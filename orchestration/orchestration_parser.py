"""
Orchestration Parser - Parse canvas workflow JSON into AgentForge workflow
Converts canvas workflow JSON to executable AgentForge orchestration
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque


class OrchestrationParser:
    """Parse canvas workflow JSON and convert to AgentForge format"""
    
    def __init__(self):
        """Initialize parser"""
        pass
    
    def parse_workflow(self, canvas_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse canvas workflow JSON into AgentForge workflow structure.
        
        Args:
            canvas_json: Canvas workflow JSON with structure:
                {
                    "version": "1.1.0",
                    "format": "website-playground",
                    "name": "Workflow Name",
                    "notes": "Description",
                    "createdAt": "ISO timestamp",
                    "graph": {
                        "cells": [nodes and links]
                    }
                }
            
        Returns:
            Dictionary with parsed workflow data:
            {
                "name": workflow name,
                "notes": workflow notes,
                "version": workflow version,
                "format": workflow format,
                "createdAt": creation timestamp,
                "nodes": {node_id: node_data},
                "links": [link_data],
                "execution_order": [node_ids],
                "agent_sequence": [agent_configs]
            }
        """
        try:
            # Extract workflow metadata
            workflow_name = canvas_json.get("name", "Untitled Workflow")
            workflow_notes = canvas_json.get("notes", "")
            workflow_version = canvas_json.get("version", "1.0.0")
            workflow_format = canvas_json.get("format", "unknown")
            created_at = canvas_json.get("createdAt", "")
            
            # Extract graph data
            graph = canvas_json.get("graph", {})
            cells = graph.get("cells", [])
            
            # Parse nodes and links
            nodes = {}
            links = []
            
            for cell in cells:
                cell_type = cell.get("type", "")
                
                # Parse node (AgentNode, CircleNode, etc.)
                if cell_type.startswith("custom."):
                    node_id = cell.get("id")
                    if not node_id:
                        continue
                    
                    # Extract node data
                    attrs = cell.get("attrs", {})
                    title = attrs.get("title", {}).get("text", "")
                    details = attrs.get("details", {}).get("text", "")
                    config = cell.get("config", {})
                    template_id = cell.get("templateId", "")
                    
                    # Parse details text into config (backward compatibility)
                    if details and not config:
                        parsed_config = self._parse_details_text(details)
                        config.update(parsed_config)
                    
                    # Build node data
                    node_data = {
                        "id": node_id,
                        "type": cell_type,
                        "template": template_id,
                        "title": title,
                        "config": config,
                        "position": cell.get("position", {}),
                        "size": cell.get("size", {})
                    }
                    
                    nodes[node_id] = node_data
                
                # Parse link
                elif cell_type == "standard.Link":
                    source = cell.get("source", {})
                    target = cell.get("target", {})
                    
                    source_id = source.get("id")
                    target_id = target.get("id")
                    source_port = source.get("port", "")
                    target_port = target.get("port", "")
                    
                    if source_id and target_id:
                        links.append({
                            "source": source_id,
                            "target": target_id,
                            "source_port": source_port,
                            "target_port": target_port
                        })
            
            # Determine execution order (topological sort)
            execution_order = self._topological_sort(nodes, links)
            
            # Map nodes to agent configurations
            agent_sequence = self._map_to_agents(nodes, execution_order)
            
            return {
                "name": workflow_name,
                "notes": workflow_notes,
                "version": workflow_version,
                "format": workflow_format,
                "createdAt": created_at,
                "nodes": nodes,
                "links": links,
                "execution_order": execution_order,
                "agent_sequence": agent_sequence,
                "metadata": {
                    "node_count": len(nodes),
                    "link_count": len(links),
                    "agent_count": len(agent_sequence)
                }
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse workflow: {str(e)}")
    
    def _parse_details_text(self, details_text: str) -> Dict[str, str]:
        """
        Parse details text (key: value format) into config dictionary.
        
        Args:
            details_text: Text in format "key: value\nkey2: value2"
            
        Returns:
            Dictionary with parsed config
        """
        config = {}
        if not details_text:
            return config
        
        lines = details_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip() if len(parts) > 1 else ''
                if key:
                    config[key] = value
        
        return config
    
    def _topological_sort(self, nodes: Dict[str, Any], links: List[Dict[str, Any]]) -> List[str]:
        """
        Perform topological sort to determine execution order.
        
        Args:
            nodes: Dictionary of node_id -> node_data
            links: List of link dictionaries
            
        Returns:
            List of node IDs in execution order
        """
        # Build graph
        graph = {node_id: [] for node_id in nodes.keys()}
        in_degree = {node_id: 0 for node_id in nodes.keys()}
        
        # Add edges from links
        for link in links:
            source = link.get("source")
            target = link.get("target")
            
            if source in nodes and target in nodes:
                source_template = nodes[source].get("template", "")
                target_template = nodes[target].get("template", "")
                
                # Include edges even if source is trigger-start or target is end-sink
                # We'll filter them out later in the result
                if target_template != "end-sink":
                    if target not in graph[source]:
                        graph[source].append(target)
                        in_degree[target] += 1
        
        # Find entry nodes (nodes with no incoming edges, or trigger-start)
        queue = deque()
        for node_id, node_data in nodes.items():
            template = node_data.get("template", "")
            if in_degree[node_id] == 0 or template == "trigger-start":
                queue.append(node_id)
        
        # Topological sort
        result = []
        while queue:
            node_id = queue.popleft()
            template = nodes[node_id].get("template", "")
            
            # Skip trigger-start and end-sink in result
            if template not in ["trigger-start", "end-sink"]:
                result.append(node_id)
            
            # Process neighbors
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining nodes (cycles or disconnected)
        for node_id in nodes.keys():
            if node_id not in result:
                template = nodes[node_id].get("template", "")
                if template not in ["trigger-start", "end-sink"]:
                    result.append(node_id)
        
        return result
    
    def _map_to_agents(self, nodes: Dict[str, Any], execution_order: List[str]) -> List[Dict[str, Any]]:
        """
        Map canvas nodes to AgentForge agent configurations.
        
        Args:
            nodes: Dictionary of node data
            execution_order: List of node IDs in execution order
            
        Returns:
            List of agent configuration dictionaries with:
            {
                "name": agent name (from title),
                "role": agent role (from config),
                "config": full config dict,
                "node_id": original node ID
            }
        """
        agent_sequence = []
        
        for node_id in execution_order:
            node = nodes.get(node_id)
            if not node:
                continue
            
            template_id = node.get("template", "")
            config = node.get("config", {})
            title = node.get("title", "")
            
            # Skip trigger-start and end-sink nodes
            if template_id in ["trigger-start", "end-sink"]:
                continue
            
            # Only process AgentNode types
            if node.get("type") == "custom.AgentNode":
                # Extract agent name from title (e.g., "Flight Search Agent" -> "Flight Search Agent")
                agent_name = title.strip() if title else f"Agent_{node_id[:8]}"
                
                # Extract role from config
                role = config.get("role", "")
                if not role and title:
                    # Try to infer role from title (remove "Agent" suffix)
                    role = title.replace(" Agent", "").replace("Agent", "").strip()
                
                # Build system prompt from config
                system_prompt = self._build_system_prompt_from_config(config, title, role)
                
                agent_config = {
                    "name": agent_name,
                    "role": role or agent_name,
                    "system_prompt": system_prompt,
                    "config": config,
                    "node_id": node_id,
                    "template_id": template_id
                }
                
                agent_sequence.append(agent_config)
        
        return agent_sequence
    
    def _build_system_prompt_from_config(self, config: Dict[str, Any], title: str, role: str) -> str:
        """
        Build system prompt from agent node configuration.
        
        Args:
            config: Agent configuration dictionary
            title: Agent title
            role: Agent role
            
        Returns:
            System prompt string
        """
        prompt_parts = []
        
        # Special handling for Report Generator Agent
        if ("report" in title.lower() or "report" in role.lower() or 
            config.get("format") in ["Sheet + Chart", "CSV, JSON, Charts"] or
            "chart" in str(config.get("output", "")).lower() or
            "analyst" in role.lower()):
            return self._build_report_generator_prompt(config, title, role)
        
        # Build enhanced prompt for all agents
        prompt_parts = []
        
        # Start with role if available
        if role:
            prompt_parts.append(f"You are a {role}.")
        elif title:
            prompt_parts.append(f"You are {title}.")
        else:
            prompt_parts.append(f"You are {title or 'an AI agent'}.")
        
        # Add domain-specific guidance
        prompt_parts.append("Your goal is to provide concise, well-structured, and actionable responses.")
        prompt_parts.append("**IMPORTANT**: Keep responses focused and relevant. Avoid repetition, verbose disclaimers, or off-topic content. Maximum response: 1500 words.")
        
        # Add config details with context
        if config.get("source"):
            prompt_parts.append(f"**Data Source**: {config['source']}")
        
        if config.get("criteria"):
            prompt_parts.append(f"**Focus Areas**: {config['criteria']}")
        
        if config.get("strategy"):
            prompt_parts.append(f"**Strategy**: {config['strategy']}")
        
        if config.get("route"):
            prompt_parts.append(f"**Route/Path**: {config['route']}")
        
        if config.get("dates"):
            prompt_parts.append(f"**Timeframe**: {config['dates']}")
        
        if config.get("format"):
            prompt_parts.append(f"**Output Format**: {config['format']}")
        
        if config.get("output"):
            prompt_parts.append(f"**Output Types**: {config['output']}")
        
        # Add quality guidelines
        prompt_parts.append("\n**Response Guidelines**:")
        prompt_parts.append("- Structure your response clearly with headers and sections when appropriate")
        prompt_parts.append("- Use tables for structured data comparisons")
        prompt_parts.append("- Include relevant metrics, statistics, or key findings")
        prompt_parts.append("- Provide actionable insights or recommendations when applicable")
        prompt_parts.append("- Format numbers, dates, and data consistently")
        prompt_parts.append("- Be thorough but concise")
        
        # Build final prompt
        if prompt_parts:
            return "\n".join(prompt_parts)
        else:
            # Default enhanced prompt
            return f"""You are {title or 'an AI agent'}. 

Your role is to provide helpful, accurate, and well-structured responses based on the input provided.

**Response Guidelines**:
- Structure your response clearly with appropriate sections
- Use tables for structured data when helpful
- Include key metrics and findings
- Provide actionable recommendations when relevant
- Format information consistently and professionally"""
    
    def _build_report_generator_prompt(self, config: Dict[str, Any], title: str, role: str) -> str:
        """
        Build enhanced system prompt for Report Generator agents.
        Generic and applicable to any domain (flights, medical, cybersecurity, etc.)
        
        Args:
            config: Agent configuration dictionary
            title: Agent title
            role: Agent role
            
        Returns:
            Enhanced system prompt for report generation
        """
        prompt = f"""You are {title or 'a Report Generator Agent'}, a Data Analyst and Report Generator specialized in creating concise, well-structured reports across any domain.

Your role: {role or 'Data Analyst'}

**CRITICAL**: Keep responses focused, concise, and relevant. Avoid repetition, verbose disclaimers, or off-topic content. Maximum response length: 2000 words.

**REPORT GENERATION REQUIREMENTS:**

1. **Data Processing with NumPy (Optional)**: When working with numerical data, you can use NumPy concepts for:
   - Statistical calculations: mean, median, std, min, max, percentiles
   - Data aggregation and summarization
   - Array operations for efficient data processing
   - Generate summary statistics automatically
   - Example: "Statistical analysis shows: mean=85.3, std=12.4, median=87"
   - **Note**: You can calculate statistics manually or reference NumPy functions conceptually - the goal is accurate statistical reporting, not necessarily executing NumPy code

2. **Structured Data Tables**: Always organize data in clear markdown tables with proper headers and alignment. Use tables for:
   - Comparisons (options, alternatives, results)
   - Metrics and statistics (include NumPy-calculated stats when applicable)
   - Categorized data
   - Time-series data
   - Statistical summaries (mean, median, std dev, etc.)
   - Any structured information

3. **Visual Charts**: Create ASCII/Unicode charts and visualizations using characters like:
   - Bar charts: █ ▓ ▒ ░ (for comparisons, distributions, rankings)
   - Line charts: ─ │ ┌ ┐ └ ┘ (for trends, timelines, progressions)
   - Progress indicators: ▰ ▱ (for completion, percentages)
   - Examples:
     * Comparison Chart:
       Option A:  ████████████ 75% (mean: 75, std: 5.2)
       Option B:  ████████████████████ 95% (mean: 95, std: 3.1)
       Option C:  ████████ 50% (mean: 50, std: 8.7)
     
     * Trend Chart:
       Q1 ──────● (value: 100, change: +5%)
       Q2 ──────────● (value: 120, change: +20%)
       Q3 ────────────────● (value: 135, change: +12.5%)
       Q4 ──────────────────────● (value: 150, change: +11.1%)

4. **Data Formats**: Provide data in multiple formats when applicable:
   - **Markdown Tables**: For easy reading and documentation
   - **JSON Structure**: For programmatic access and API integration (include NumPy arrays as lists)
   - **CSV Format**: For spreadsheet import and data analysis tools
   - **NumPy Arrays**: When providing code examples, show NumPy usage

5. **Report Sections**: Structure your reports with clear sections:
   - **Executive Summary**: High-level overview and key findings
   - **Statistical Summary**: NumPy-calculated statistics (mean, median, std, percentiles, etc.)
   - **Detailed Analysis**: In-depth examination of data/results
   - **Visual Comparisons**: Charts and graphs for quick understanding
   - **Data Tables**: Structured data presentation with statistics
   - **Key Metrics**: Important statistics and numbers (highlight NumPy-calculated values)
   - **Recommendations**: Actionable insights and next steps
   - **Export Formats**: JSON/CSV when applicable for further processing

6. **Statistical Calculations (NumPy-style, but flexible)**:
   - Calculate mean/average values
   - Find median values
   - Compute standard deviation
   - Identify min/max values
   - Calculate percentiles (25th, 75th, 90th, etc.)
   - Sum totals
   - Calculate variance
   - Determine correlations (when comparing multiple metrics)
   - **Approach**: You can use NumPy concepts or calculate manually - focus on providing accurate statistics in your reports

7. **Chart Types by Domain**:
   - **Business/Finance**: Price comparisons, cost analysis, ROI charts, budget breakdowns (with NumPy stats)
   - **Medical/Health**: Symptom distributions, treatment comparisons, timeline visualizations (with statistical analysis)
   - **Cybersecurity**: Threat levels, risk assessments, vulnerability rankings, incident timelines (with risk scores)
   - **Research/Analytics**: Data distributions, trend analysis, correlation visualizations (with NumPy calculations)
   - **Operations**: Process flows, performance metrics, efficiency comparisons (with statistical summaries)
   - **Any Domain**: Adapt chart types to the data context with appropriate statistics

8. **Output Quality Standards**: 
   - Use clear section headers (##, ###) for navigation
   - Format numbers consistently (currency, percentages, decimals as appropriate)
   - Include NumPy-calculated statistics in tables and summaries
   - Highlight key findings with bold text or emphasis
   - Provide actionable recommendations based on statistical analysis
   - Include summary statistics (totals, averages, ranges, std dev, percentiles, etc.)
   - Use appropriate units and labels
   - Ensure data accuracy and consistency
   - Show statistical significance when applicable

9. **Domain Adaptation**: 
   - Analyze the input data to understand the domain context
   - Extract relevant metrics based on the domain (prices for commerce, scores for assessments, counts for analytics, etc.)
   - Use NumPy to calculate domain-appropriate statistics
   - Create appropriate visualizations for the data type with statistical annotations
   - Use domain-appropriate terminology and formatting
   - Structure recommendations relevant to the domain with statistical backing

**IMPORTANT**: When processing any input data:
- Extract all key metrics, values, and statistics
- Calculate statistical summaries (mean, median, std, etc.) - use standard statistical methods
- Identify comparison opportunities (options, alternatives, before/after, etc.)
- Create appropriate visual charts for the data type with statistical annotations
- Organize information in clear, scannable tables with calculated statistics
- Provide JSON/CSV export formats when structured data is present
- Include summary statistics (mean, median, std, percentiles) - calculate accurately
- Add actionable recommendations based on statistical analysis
- Format consistently and professionally
- **Flexibility**: Use statistical calculations appropriate to the data - NumPy is optional, focus on accurate reporting

**Example Statistical Summary Format**:
```
## Statistical Summary
- **Mean**: 85.3 units
- **Median**: 87.0 units
- **Standard Deviation**: 12.4 units
- **Min/Max**: 45 / 120 units
- **25th Percentile**: 75.0 units
- **75th Percentile**: 95.0 units
- **Range**: 75 units
```

Always format your response as a comprehensive, professional report with clear structure, visual elements, statistical analysis, and export-ready data formats. Adapt your approach to the specific domain and data type you're analyzing."""

        # Add specific config details
        if config.get("format"):
            prompt += f"\n\n**Required Output Format**: {config['format']}"
        
        if config.get("output"):
            prompt += f"\n**Output Types**: {config['output']}"
        
        if config.get("source"):
            prompt += f"\n**Data Source**: {config['source']}"
        
        if config.get("strategy"):
            prompt += f"\n**Analysis Strategy**: {config['strategy']}"
        
        return prompt
    
    def validate_workflow(self, parsed_workflow: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parsed workflow.
        
        Args:
            parsed_workflow: Parsed workflow dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for nodes
        if not parsed_workflow.get("nodes"):
            errors.append("Workflow has no nodes")
        
        # Check for agent sequence
        agent_sequence = parsed_workflow.get("agent_sequence", [])
        if not agent_sequence:
            errors.append("Workflow has no agents to execute")
        
        # Check for execution order
        execution_order = parsed_workflow.get("execution_order", [])
        if not execution_order:
            errors.append("Could not determine execution order")
        
        return len(errors) == 0, errors

