"""
Workflow Executor - Execute parsed workflows using AgentForge agents
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .orchestration_parser import OrchestrationParser
from src.agent_manager import AgentManager
from src.custom_agent import CustomAgent
from src.logger_config import get_logger

logger = get_logger("agentforge.workflow_executor")

# Pre-compile commonly used regex patterns for better performance
_URL_PATTERN = re.compile(r'https?://[^\s]+')
_CODE_BLOCK_PATTERN = re.compile(r'```[\w]*\n.*?```|```.*?```', re.DOTALL | re.MULTILINE)
_METADATA_PATTERNS = [
    re.compile(r'Data source:.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Strategy:.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Output format:.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Focus on:.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Route:.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Dates:.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'IMPORTANT:.*?(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Your response should.*?(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Do NOT include.*?(?=\n\n|\n[A-Z]|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
    re.compile(r'Be direct.*?(?=\n|$)', re.IGNORECASE | re.DOTALL | re.MULTILINE),
]


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
                            
                            # Clean response: remove URLs and repetitive content
                            import re
                            # Remove URLs
                            response = re.sub(r'https?://[^\s]+', '', response)
                            
                            # Remove repetitive phrases (catch long repetitive strings)
                            lines = response.split('\n')
                            seen_phrases = set()
                            unique_lines = []
                            for line in lines:
                                line_stripped = line.strip()
                                if not line_stripped:
                                    continue
                                # For long lines, check if we've seen similar content
                                if len(line_stripped) > 50:
                                    key_phrase = line_stripped[:80] if len(line_stripped) > 80 else line_stripped
                                    if key_phrase.lower() in seen_phrases:
                                        continue  # Skip repetitive long lines
                                    seen_phrases.add(key_phrase.lower())
                                unique_lines.append(line)
                            response = '\n'.join(unique_lines)
                            
                            # Extract key data points (numbers, prices, etc.) for report generators
                            if "report" in agent_config.get("name", "").lower() or "analyst" in agent_config.get("role", "").lower():
                                # For report generators, extract structured data
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
                
                # Helper functions for cleaning (defined once per agent iteration)
                import re
                
                def clean_context_for_agent(ctx):
                    """Remove all metadata and guidelines from context before passing to agent"""
                    if not ctx:
                        return ""
                    # Use pre-compiled patterns for better performance
                    cleaned = ctx
                    for pattern in _METADATA_PATTERNS:
                        cleaned = pattern.sub('', cleaned)
                    return cleaned.strip()
                
                def clean_response_text(text):
                    """Remove system prompts, guidelines, agent metadata, and code blocks from responses"""
                    if not text:
                        return ""
                    
                    # Remove URLs first (they're often repetitive and long) - use pre-compiled pattern
                    cleaned = _URL_PATTERN.sub('', text)
                    
                    # Remove code blocks - use pre-compiled pattern
                    cleaned = _CODE_BLOCK_PATTERN.sub('', cleaned)
                    
                    # Remove common system prompt patterns (more aggressive)
                    patterns_to_remove = [
                        r'Response Guidelines:.*?(?=\n\n|\n[A-Z]|$)',
                        r'Data source:.*?(?=\n|$)',
                        r'Strategy:.*?(?=\n|$)',
                        r'Output format:.*?(?=\n|$)',
                        r'Output types:.*?(?=\n|$)',
                        r'Focus on:.*?(?=\n|$)',
                        r'Route:.*?(?=\n|$)',
                        r'Dates:.*?(?=\n|$)',
                        r'Context:.*?(?=\n\n|$)',
                        r'## Previous Results:.*?(?=\n\n|$)',
                        r'\[.*?Agent\]:\s*',
                        r'Human:.*?(?=\n|$)',
                        r'\*\*STRICT RULES\*\*:.*?(?=\n\n|$)',
                        r'\*\*IMPORTANT\*\*:.*?(?=\n\n|$)',
                        r'IMPORTANT:.*?(?=\n\n|\n[A-Z]|$)',
                        r'CRITICAL:.*?(?=\n\n|$)',
                        r'Your response should.*?(?=\n\n|$)',
                        r'Do NOT include.*?(?=\n\n|$)',
                        r'Maximum \d+ words.*?(?=\n|$)',
                        r'Be direct.*?(?=\n|$)',
                        r'NO repetition.*?(?=\n|$)',
                    ]
                    for pattern in patterns_to_remove:
                        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    
                    # Remove lines that are just guidelines/metadata or code
                    lines = cleaned.split('\n')
                    filtered_lines = []
                    skip_patterns = [
                        r'^Response Guidelines',
                        r'^Data source',
                        r'^Strategy',
                        r'^Output format',
                        r'^Output types',
                        r'^Focus on',
                        r'^Route:',
                        r'^Dates:',
                        r'^Context:',
                        r'^Human:',
                        r'^IMPORTANT:',
                        r'^Your response',
                        r'^Do NOT',
                        r'^Maximum \d+',
                        r'^Be direct',
                        r'^NO ',
                    ]
                    # Patterns to detect code lines
                    code_line_patterns = [
                        r'^import\s+',
                        r'^from\s+\w+\s+import',
                        r'^def\s+\w+\s*\(',
                        r'^class\s+\w+',
                        r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]',  # Variable assignment
                        r'^\s*print\s*\(',
                        r'^\s*return\s+',
                        r'^\s*if\s+.*:',
                        r'^\s*for\s+.*:',
                        r'^\s*while\s+.*:',
                        r'^\s*#.*',  # Comments
                    ]
                    
                    for line in lines:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                        
                        # Skip if line matches skip patterns
                        should_skip = False
                        for pattern in skip_patterns:
                            if re.match(pattern, line_stripped, re.IGNORECASE):
                                should_skip = True
                                break
                        
                        # Skip if line looks like code
                        if not should_skip:
                            for code_pattern in code_line_patterns:
                                if re.match(code_pattern, line_stripped):
                                    should_skip = True
                                    break
                        
                        if not should_skip:
                            filtered_lines.append(line)
                    
                    return '\n'.join(filtered_lines).strip()
                
                # Clean context before passing to agent
                agent_context_clean = clean_context_for_agent(agent_context)
                
                # Process with agent
                try:
                    result_dict = agent.process(agent_input, context=agent_context_clean)
                    result_dict["step"] = idx + 1
                    
                    # Clean response immediately after generation
                    response_raw = result_dict.get('response', '')
                    if response_raw:
                        result_dict['response'] = clean_response_text(response_raw)
                    
                    # Post-process report generator responses to ensure charts are included
                    if ("report" in agent_name.lower() or "analyst" in agent_config.get("role", "").lower() or 
                        "chart" in str(agent_config.get("config", {}).get("output", "")).lower()):
                        response = result_dict.get('response', '')
                        # Check if response has visual charts (bar characters)
                        has_charts = any(char in response for char in ['█', '▓', '▒', '░', '─', '│', '▰', '▱'])
                        
                        if not has_charts and response:
                            # Generic data extraction from ALL previous agent outputs
                            # Similar to JavaScript extractChartData function - works for ANY scenario
                            # clean_response_text is already defined above
                            
                            # Collect and clean all text from current and previous responses
                            all_text = clean_response_text(response)
                            for prev_result in results:
                                cleaned_prev = clean_response_text(prev_result.get('response', ''))
                                all_text += " " + cleaned_prev
                            
                            # Filter words to exclude (agent names, system terms, noise)
                            exclude_words = {
                                'agent', 'search', 'finder', 'generator', 'discount', 'flight', 'report',
                                'response', 'guidelines', 'data', 'source', 'strategy', 'output', 'format',
                                'context', 'previous', 'results', 'human', 'focus', 'route', 'dates',
                                'item', 'option', 'key', 'full', 'details', 'points', 'summary'
                            }
                            
                            def is_valid_label(label):
                                """Check if label is meaningful and not noise"""
                                if not label or len(label) < 2 or len(label) > 30:
                                    return False
                                label_lower = label.lower()
                                # Check if contains excluded words
                                words = label_lower.split()
                                if any(w in exclude_words for w in words):
                                    return False
                                # Check if it's just a number or generic term
                                if label_lower in ['item', 'option', 'value', 'data', 'result']:
                                    return False
                                return True
                            
                            # Extract data points with labels (generic pattern extraction)
                            data_points = []
                            
                            # Pattern 1: Numbers with currency symbols (£$€)
                            for match in re.finditer(r'([£$€])\s*(\d+\.?\d*)', all_text):
                                currency = match.group(1)
                                value = float(match.group(2))
                                # Extract label from context (80 chars before and after)
                                context_start = max(0, match.start() - 80)
                                context_end = min(len(all_text), match.end() + 30)
                                context = all_text[context_start:context_end]
                                label = _extract_label_from_context(context)
                                if label and is_valid_label(label):
                                    data_points.append({'label': label, 'value': value, 'unit': currency})
                            
                            # Pattern 2: Numbers with percentages
                            for match in re.finditer(r'(\d+\.?\d*)\s*%', all_text):
                                value = float(match.group(1))
                                if value < 0 or value > 100:
                                    continue
                                context_start = max(0, match.start() - 80)
                                context_end = min(len(all_text), match.end() + 30)
                                context = all_text[context_start:context_end]
                                label = _extract_label_from_context(context)
                                if label and is_valid_label(label):
                                    data_points.append({'label': label, 'value': value, 'unit': '%'})
                            
                            # Pattern 3: Colon-separated labels (e.g., "Price: 150", "Score: 8.5", "British Airways: £450")
                            # More flexible pattern to catch various formats
                            for match in re.finditer(r'([A-Z][A-Za-z\s]{2,40}):\s*([£$€]?\s*\d+\.?\d*)', all_text):
                                label = match.group(1).strip()
                                value_str = match.group(2).strip()
                                # Extract number and currency
                                num_match = re.search(r'(\d+\.?\d*)', value_str)
                                if not num_match:
                                    continue
                                value = float(num_match.group(1))
                                if is_valid_label(label) and value > 0:
                                    unit = ''
                                    # Check if there's a currency symbol in the value
                                    if re.search(r'[£$€]', value_str):
                                        unit = re.search(r'([£$€])', value_str).group(1)
                                    elif '%' in value_str:
                                        unit = '%'
                                    data_points.append({'label': label, 'value': value, 'unit': unit})
                            
                            # Pattern 4: Table structures (markdown tables)
                            table_pattern = r'\|([^|]+)\|([^|]+)\|'
                            for match in re.finditer(table_pattern, all_text):
                                label_cell = match.group(1).strip()
                                value_cell = match.group(2).strip()
                                # Extract number from value cell
                                num_match = re.search(r'(\d+\.?\d*)', value_cell)
                                if num_match and is_valid_label(label_cell):
                                    value = float(num_match.group(1))
                                    unit = ''
                                    if re.search(r'[£$€]', value_cell):
                                        unit = re.search(r'([£$€])', value_cell).group(1)
                                    elif '%' in value_cell:
                                        unit = '%'
                                    if value > 0:
                                        data_points.append({'label': label_cell, 'value': value, 'unit': unit})
                            
                            # Pattern 5: Look for structured data patterns (e.g., "from £450", "costs $120", "rated 8.5")
                            structured_patterns = [
                                (r'(?:from|costs?|priced?|pays?|worth)\s+([£$€])\s*(\d+\.?\d*)', 2, 1),  # "from £450" - group 1=currency, group 2=number
                                (r'([£$€])\s*(\d+\.?\d*)\s+(?:for|each|per)', 2, 1),  # "£450 for" - group 1=currency, group 2=number
                                (r'rated\s+(\d+\.?\d*)\s*(?:out of|/|\%)', 1, None),  # "rated 8.5/10" - group 1=number
                                (r'(\d+\.?\d*)\s*(?:hours?|hrs?|minutes?|mins?)\s+(?:duration|flight|trip)', 1, None),  # "2.5 hours duration" - group 1=number
                            ]
                            for pattern, value_group, unit_group in structured_patterns:
                                for match in re.finditer(pattern, all_text, re.IGNORECASE):
                                    try:
                                        # value_group is the group number containing the numeric value
                                        value_str = match.group(value_group)
                                        value = float(value_str)
                                        
                                        # unit_group is the group number containing the unit (currency, etc.)
                                        unit = ''
                                        if unit_group:
                                            unit = match.group(unit_group)
                                        
                                        context_start = max(0, match.start() - 60)
                                        context_end = min(len(all_text), match.end() + 40)
                                        context = all_text[context_start:context_end]
                                        label = _extract_label_from_context(context)
                                        if label and is_valid_label(label) and value > 0:
                                            data_points.append({'label': label, 'value': value, 'unit': unit})
                                    except (ValueError, IndexError):
                                        # Skip if conversion fails or group doesn't exist
                                        continue
                            
                            # Pattern 6: Generic numbers with meaningful context (fallback - only if we still don't have enough)
                            if len(data_points) < 2:
                                for match in re.finditer(r'\b(\d+\.?\d*)\b', all_text):
                                    value = float(match.group(1))
                                    # Skip years, small numbers, or unrealistic values
                                    if value < 0.1 or value > 1000000 or (1900 < value < 2100 and value % 1 == 0):
                                        continue
                                    context_start = max(0, match.start() - 100)
                                    context_end = min(len(all_text), match.end() + 50)
                                    context = all_text[context_start:context_end]
                                    label = _extract_label_from_context(context)
                                    if label and is_valid_label(label) and len(data_points) < 10:
                                        unit = ''
                                        if re.search(r'[£$€]', context):
                                            unit = re.search(r'([£$€])', context).group(1)
                                        data_points.append({'label': label, 'value': value, 'unit': unit})
                            
                            # Remove duplicates and filter out invalid labels
                            unique_points = []
                            seen = set()
                            for point in data_points:
                                # Normalize label for deduplication
                                label_normalized = point['label'].lower().strip()
                                key = (label_normalized, round(point['value'], 2))
                                if key not in seen and point['value'] > 0 and is_valid_label(point['label']):
                                    seen.add(key)
                                    unique_points.append(point)
                                    if len(unique_points) >= 8:
                                        break
                            
                            # Generate chart and table if we have meaningful data
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
