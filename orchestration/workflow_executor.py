"""
Workflow Executor - Execute parsed workflows using AgentForge agents
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .orchestration_parser import OrchestrationParser
from src.agent_manager import AgentManager
from src.custom_agent import CustomAgent


def _extract_label_from_context(context: str) -> Optional[str]:
    """
    Extract a meaningful label from context text.
    Similar to JavaScript extractContext function - works for any scenario.
    """
    import re
    
    if not context:
        return None
    
    # Common stop words to filter out
    stop_words = {'The', 'This', 'That', 'These', 'Those', 'With', 'From', 'To', 'And', 'Or', 'But', 
                  'For', 'In', 'On', 'At', 'By', 'As', 'Is', 'Are', 'Was', 'Were', 'Has', 'Have', 
                  'Had', 'Will', 'Would', 'Should', 'Could', 'May', 'Might', 'Can', 'Must'}
    
    # Pattern 1: Colon-separated labels (e.g., "Price: 150", "Score: 8.5")
    colon_match = re.search(r'([A-Z][A-Za-z\s]{1,25}):\s*\d+', context)
    if colon_match:
        label = colon_match.group(1).strip()
        if label and label.split()[0] not in stop_words and 2 < len(label) < 30:
            return label
    
    # Pattern 2: Quoted strings (e.g., "London to Paris", "Product Name")
    quoted_match = re.search(r'"([^"]{3,30})"', context)
    if quoted_match:
        label = quoted_match.group(1).strip()
        if 2 < len(label) < 30:
            return label
    
    # Pattern 3: Capitalized phrases (e.g., "Flight Price", "Hotel Rating")
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', context)
    if capitalized:
        for phrase in capitalized:
            words = phrase.split()
            if not any(w in stop_words for w in words) and 3 < len(phrase) < 30:
                return phrase
    
    # Pattern 4: Table-like structures (e.g., "| Item | Price |")
    table_match = re.search(r'\|\s*([A-Za-z\s]{2,25})\s*\|', context)
    if table_match:
        label = table_match.group(1).strip()
        if label.split()[0] not in stop_words and 2 < len(label) < 30:
            return label
    
    # Pattern 5: Route patterns (e.g., "London to Paris", "NYC -> LA")
    route_match = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:to|->|→|-)\s+([A-Z][A-Za-z]+)', context, re.IGNORECASE)
    if route_match:
        return f"{route_match.group(1)} → {route_match.group(2)}"
    
    return None


class WorkflowExecutor:
    """Execute canvas workflows using AgentForge agents"""
    
    def __init__(self, agent_manager: AgentManager, llm=None, auto_create_agents: bool = True):
        """
        Initialize workflow executor.
        
        Args:
            agent_manager: AgentManager instance
            llm: Language model instance
            auto_create_agents: If True, create agents that don't exist from workflow config
        """
        self.agent_manager = agent_manager
        self.llm = llm
        self.auto_create_agents = auto_create_agents
        self.parser = OrchestrationParser()
    
    def execute_workflow(
        self,
        canvas_json: Dict[str, Any],
        initial_input: str,
        context: str = "",
        pass_mode: str = "cumulative"
    ) -> Dict[str, Any]:
        """
        Execute a canvas workflow.
        
        Args:
            canvas_json: Canvas workflow JSON
            initial_input: Initial input text
            context: Additional context
            pass_mode: How to pass data between agents
                - "cumulative": Each agent sees all previous outputs
                - "sequential": Each agent only sees previous agent's output
                - "parallel": All agents see only initial input
        
        Returns:
            Dictionary with execution results
        """
        try:
            # Parse workflow
            parsed_workflow = self.parser.parse_workflow(canvas_json)
            
            # Validate workflow
            is_valid, errors = self.parser.validate_workflow(parsed_workflow)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Workflow validation failed: {', '.join(errors)}",
                    "workflow_name": parsed_workflow.get("name", "Unknown")
                }
            
            # Get agent sequence
            agent_sequence = parsed_workflow.get("agent_sequence", [])
            if not agent_sequence:
                return {
                    "success": False,
                    "error": "No agents found in workflow",
                    "workflow_name": parsed_workflow.get("name", "Unknown")
                }
            
            # Ensure all agents exist
            missing_agents = []
            for agent_config in agent_sequence:
                agent_name = agent_config.get("name")
                if not self.agent_manager.agent_exists(agent_name):
                    if self.auto_create_agents:
                        # Create agent from config
                        agent = self._create_agent_from_config(agent_config)
                        if agent:
                            self.agent_manager.save_agent(agent)
                        else:
                            missing_agents.append(agent_name)
                    else:
                        missing_agents.append(agent_name)
            
            if missing_agents:
                return {
                    "success": False,
                    "error": f"Agents not found and could not be created: {', '.join(missing_agents)}",
                    "workflow_name": parsed_workflow.get("name", "Unknown")
                }
            
            # Execute agents in sequence
            results = []
            current_input = initial_input
            all_outputs = []
            
            for idx, agent_config in enumerate(agent_sequence):
                agent_name = agent_config.get("name")
                
                # Load agent
                agent = self.agent_manager.load_agent(agent_name, llm=self.llm)
                if not agent:
                    results.append({
                        "agent": agent_name,
                        "role": agent_config.get("role", "Unknown"),
                        "response": f"Error: Could not load agent '{agent_name}'",
                        "success": False,
                        "step": idx + 1
                    })
                    continue
                
                # Determine input based on pass mode
                if pass_mode == "cumulative" and idx > 0:
                    # Include all previous outputs with better formatting
                    # Truncate each response to avoid overwhelming the model
                    previous_outputs = []
                    for r in results:
                        if r.get('success', False):
                            agent_name = r.get('agent', 'Unknown')
                            response = r.get('response', '')
                            
                            # Extract key data points (numbers, prices, etc.) for report generators
                            if "report" in agent_config.get("name", "").lower() or "analyst" in agent_config.get("role", "").lower():
                                # For report generators, extract structured data
                                import re
                                # Extract prices, numbers, percentages
                                prices = re.findall(r'[£$€]\s*[\d,]+', response)
                                numbers = re.findall(r'\d+\.?\d*\s*(?:hours?|minutes?|%|units?)', response, re.IGNORECASE)
                                key_data = prices[:5] + numbers[:5]  # Top 5 of each
                                if key_data:
                                    response = f"Key data points: {', '.join(key_data[:8])}\n\nFull details: {response[:300]}"
                            
                            # Truncate each response to max 400 chars to keep context manageable
                            if len(response) > 400:
                                # Try to cut at sentence boundary
                                truncated = response[:400]
                                last_period = truncated.rfind('.')
                                last_newline = truncated.rfind('\n')
                                cut_point = max(last_period, last_newline)
                                if cut_point > 300:
                                    response = response[:cut_point + 1]
                                else:
                                    response = truncated
                            
                            previous_outputs.append(f"**{agent_name}**: {response}")
                    
                    chain_context = "## Previous Results:\n\n" + "\n\n".join(previous_outputs)
                    # Limit total context length more aggressively
                    max_context = 1500
                    if len(chain_context) > max_context:
                        chain_context = chain_context[:max_context] + "\n[Previous results truncated]"
                    
                    agent_input = initial_input
                    agent_context = f"{context}\n\n{chain_context}" if context else chain_context
                    
                    # Final truncation of total context
                    if len(agent_context) > 2000:
                        agent_context = agent_context[:2000] + "\n[Context truncated]"
                
                elif pass_mode == "sequential" and idx > 0:
                    # Only use previous agent's output, but include original context
                    prev_response = results[-1]['response'] if results else ""
                    agent_input = prev_response
                    agent_context = f"{context}\n\n**Original Request:** {initial_input}" if context else f"**Original Request:** {initial_input}"
                
                else:  # First agent or parallel mode
                    agent_input = current_input
                    agent_context = context
                
                # Process with agent
                try:
                    result_dict = agent.process(agent_input, context=agent_context)
                    result_dict["step"] = idx + 1
                    
                    # Post-process report generator responses to ensure charts are included
                    if ("report" in agent_name.lower() or "analyst" in agent_config.get("role", "").lower() or 
                        "chart" in str(agent_config.get("config", {}).get("output", "")).lower()):
                        response = result_dict.get('response', '')
                        # Check if response has visual charts (bar characters)
                        has_charts = any(char in response for char in ['█', '▓', '▒', '░', '─', '│', '▰', '▱'])
                        
                        if not has_charts and response:
                            # Generic data extraction from ALL previous agent outputs
                            # Similar to JavaScript extractChartData function - works for ANY scenario
                            import re
                            
                            # Collect all text from current and previous responses
                            all_text = response
                            for prev_result in results:
                                all_text += " " + prev_result.get('response', '')
                            
                            # Extract data points with labels (generic pattern extraction)
                            data_points = []
                            
                            # Pattern 1: Numbers with currency symbols
                            for match in re.finditer(r'([£$€])\s*(\d+\.?\d*)', all_text):
                                currency = match.group(1)
                                value = float(match.group(2))
                                # Extract label from context (50 chars before)
                                context_start = max(0, match.start() - 50)
                                context = all_text[context_start:match.start()]
                                label = _extract_label_from_context(context)
                                if not label:
                                    label = f"Item {len(data_points) + 1}"
                                data_points.append({'label': label, 'value': value, 'unit': currency})
                            
                            # Pattern 2: Numbers with percentages
                            for match in re.finditer(r'(\d+\.?\d*)\s*%', all_text):
                                value = float(match.group(1))
                                context_start = max(0, match.start() - 50)
                                context = all_text[context_start:match.start()]
                                label = _extract_label_from_context(context) or f"Item {len(data_points) + 1}"
                                data_points.append({'label': label, 'value': value, 'unit': '%'})
                            
                            # Pattern 3: Colon-separated labels (e.g., "Price: 150", "Score: 8.5")
                            for match in re.finditer(r'([A-Z][A-Za-z\s]{2,30}):\s*(\d+\.?\d*)', all_text):
                                label = match.group(1).strip()
                                value = float(match.group(2))
                                # Skip common stop words
                                stop_words = {'The', 'This', 'That', 'These', 'With', 'From', 'To', 'And'}
                                if label.split()[0] not in stop_words:
                                    data_points.append({'label': label, 'value': value, 'unit': ''})
                            
                            # Pattern 4: Generic numbers with context (fallback)
                            if len(data_points) < 2:
                                for match in re.finditer(r'\b(\d+\.?\d*)\b', all_text):
                                    value = float(match.group(1))
                                    # Skip years, small numbers, or unrealistic values
                                    if value < 0.1 or value > 1000000 or (1900 < value < 2100 and value % 1 == 0):
                                        continue
                                    context_start = max(0, match.start() - 80)
                                    context_end = min(len(all_text), match.end() + 30)
                                    context = all_text[context_start:context_end]
                                    label = _extract_label_from_context(context)
                                    if label and len(data_points) < 10:
                                        data_points.append({'label': label, 'value': value, 'unit': ''})
                            
                            # Remove duplicates and limit to top 8
                            unique_points = []
                            seen = set()
                            for point in data_points:
                                key = (point['label'], round(point['value'], 2))
                                if key not in seen and point['value'] > 0:
                                    seen.add(key)
                                    unique_points.append(point)
                                    if len(unique_points) >= 8:
                                        break
                            
                            # Generate chart and table if we have data
                            if len(unique_points) >= 2:
                                # Sort by value (descending)
                                unique_points.sort(key=lambda x: x['value'], reverse=True)
                                
                                # Determine unit and title from data
                                units = [p['unit'] for p in unique_points if p['unit']]
                                common_unit = max(set(units), key=units.count) if units else ''
                                
                                # Create table
                                table_section = "## Data Summary\n\n"
                                table_section += "| Item | Value |\n"
                                table_section += "|------|-------|\n"
                                for point in unique_points:
                                    value_str = f"{point['value']:.1f}{point['unit']}" if point['unit'] else f"{point['value']:.1f}"
                                    table_section += f"| {point['label'][:30]} | {value_str} |\n"
                                table_section += "\n"
                                
                                # Create chart
                                max_value = max(p['value'] for p in unique_points)
                                chart_section = "## Comparison Chart\n```\n"
                                for point in unique_points:
                                    bar_length = int((point['value'] / max_value) * 20) if max_value > 0 else 0
                                    bar = '█' * max(1, bar_length)
                                    value_str = f"{point['value']:.1f}{point['unit']}" if point['unit'] else f"{point['value']:.1f}"
                                    chart_section += f"{point['label'][:25]:25s} {bar} {value_str}\n"
                                chart_section += "```\n\n"
                                
                                # Add recommendation
                                best = unique_points[0]
                                rec_section = f"## Recommendation\nBest option: {best['label']} with value {best['value']:.1f}{best['unit']}.\n"
                                
                                result_dict['response'] = response + table_section + chart_section + rec_section
                    
                    results.append(result_dict)
                    
                    if result_dict.get("success"):
                        all_outputs.append(f"[{agent_name}]: {result_dict.get('response', '')}")
                
                except Exception as e:
                    results.append({
                        "agent": agent_name,
                        "role": agent_config.get("role", "Unknown"),
                        "response": f"Error during execution: {str(e)}",
                        "success": False,
                        "step": idx + 1
                    })
            
            # Build final output
            successful_steps = sum(1 for r in results if r.get("success", False))
            final_output = "\n\n".join(all_outputs) if all_outputs else "No successful outputs"
            
            return {
                "success": successful_steps > 0,
                "workflow_name": parsed_workflow.get("name", "Unknown"),
                "workflow_notes": parsed_workflow.get("notes", ""),
                "total_steps": len(agent_sequence),
                "successful_steps": successful_steps,
                "results": results,
                "final_output": final_output,
                "metadata": parsed_workflow.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "workflow_name": canvas_json.get("name", "Unknown")
            }
    
    def _create_agent_from_config(self, agent_config: Dict[str, Any]) -> Optional[CustomAgent]:
        """
        Create a CustomAgent from workflow agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            CustomAgent instance or None if creation fails
        """
        try:
            name = agent_config.get("name", "Unknown Agent")
            role = agent_config.get("role", name)
            system_prompt = agent_config.get("system_prompt", "")
            
            # If no system prompt, create a basic one
            if not system_prompt:
                system_prompt = f"You are {name}. {role}. Provide helpful, accurate responses based on the input provided."
            
            # Create agent
            agent = CustomAgent(
                name=name,
                role=role,
                system_prompt=system_prompt,
                llm=self.llm
            )
            
            return agent
            
        except Exception as e:
            print(f"Error creating agent from config: {e}")
            return None
