"""
Workflow Templates - Pre-built canvas workflow templates
Ready-to-use workflows for common use cases
"""

from typing import List, Dict, Any


class WorkflowTemplates:
    """Pre-built canvas workflow templates"""
    
    @staticmethod
    def get_all_templates() -> List[Dict[str, Any]]:
        """Get all workflow templates"""
        return [
            {
                "name": "Content Creation Pipeline",
                "description": "Brainstorm → Write → Review → Polish",
                "category": "Content",
                "workflow": {
                    "version": "1.1.0",
                    "format": "website-playground",
                    "name": "Content Creation Pipeline",
                    "notes": "Complete content creation workflow from ideation to final polish",
                    "graph": {
                        "cells": [
                            {
                                "type": "custom.CircleNode",
                                "id": "start-1",
                                "templateId": "trigger-start",
                                "attrs": {
                                    "title": {"text": "Start"},
                                    "details": {"text": "Content request"}
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-1",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Brainstorming Assistant"},
                                    "details": {"text": "Generate ideas"}
                                },
                                "config": {
                                    "role": "Brainstorming Assistant",
                                    "system_prompt": "Generate creative ideas and concepts"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-2",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Creative Writer"},
                                    "details": {"text": "Write content"}
                                },
                                "config": {
                                    "role": "Creative Writer",
                                    "system_prompt": "Write engaging and creative content"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-3",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Code Reviewer"},
                                    "details": {"text": "Review and edit"}
                                },
                                "config": {
                                    "role": "Code Reviewer",
                                    "system_prompt": "Review and improve content quality"
                                }
                            },
                            {
                                "type": "custom.CircleNode",
                                "id": "end-1",
                                "templateId": "end-sink",
                                "attrs": {
                                    "title": {"text": "End"},
                                    "details": {"text": "Final content"}
                                }
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "start-1", "port": "start-1-out"},
                                "target": {"id": "agent-1", "port": "agent-1-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-1", "port": "agent-1-out"},
                                "target": {"id": "agent-2", "port": "agent-2-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-2", "port": "agent-2-out"},
                                "target": {"id": "agent-3", "port": "agent-3-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-3", "port": "agent-3-out"},
                                "target": {"id": "end-1", "port": "end-1-in"}
                            }
                        ]
                    }
                }
            },
            {
                "name": "Research & Analysis",
                "description": "Research → Analyze → Report",
                "category": "Research",
                "workflow": {
                    "version": "1.1.0",
                    "format": "website-playground",
                    "name": "Research & Analysis",
                    "notes": "Complete research workflow with analysis and reporting",
                    "graph": {
                        "cells": [
                            {
                                "type": "custom.CircleNode",
                                "id": "start-1",
                                "templateId": "trigger-start",
                                "attrs": {
                                    "title": {"text": "Start"},
                                    "details": {"text": "Research topic"}
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-1",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Research Analyst"},
                                    "details": {"text": "Gather information"}
                                },
                                "config": {
                                    "role": "Research Analyst",
                                    "system_prompt": "Conduct thorough research on the given topic"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-2",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Data Analyst"},
                                    "details": {"text": "Analyze data"}
                                },
                                "config": {
                                    "role": "Data Analyst",
                                    "system_prompt": "Analyze research data and extract insights"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-3",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Report Generator"},
                                    "details": {"text": "Generate report"}
                                },
                                "config": {
                                    "role": "Report Generator",
                                    "system_prompt": "Create comprehensive reports with charts and tables"
                                }
                            },
                            {
                                "type": "custom.CircleNode",
                                "id": "end-1",
                                "templateId": "end-sink",
                                "attrs": {
                                    "title": {"text": "End"},
                                    "details": {"text": "Final report"}
                                }
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "start-1", "port": "start-1-out"},
                                "target": {"id": "agent-1", "port": "agent-1-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-1", "port": "agent-1-out"},
                                "target": {"id": "agent-2", "port": "agent-2-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-2", "port": "agent-2-out"},
                                "target": {"id": "agent-3", "port": "agent-3-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-3", "port": "agent-3-out"},
                                "target": {"id": "end-1", "port": "end-1-in"}
                            }
                        ]
                    }
                }
            },
            {
                "name": "Security Assessment",
                "description": "Multiple security perspectives in parallel",
                "category": "Security",
                "workflow": {
                    "version": "1.1.0",
                    "format": "website-playground",
                    "name": "Security Assessment",
                    "notes": "Parallel security analysis from multiple perspectives",
                    "graph": {
                        "cells": [
                            {
                                "type": "custom.CircleNode",
                                "id": "start-1",
                                "templateId": "trigger-start",
                                "attrs": {
                                    "title": {"text": "Start"},
                                    "details": {"text": "Security review"}
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-1",
                                "templateId": "cyber-agent",
                                "attrs": {
                                    "title": {"text": "Cybersecurity Analyst"},
                                    "details": {"text": "Security analysis"}
                                },
                                "config": {
                                    "role": "Cybersecurity Analyst",
                                    "system_prompt": "Analyze security vulnerabilities and threats"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-2",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Code Reviewer"},
                                    "details": {"text": "Code review"}
                                },
                                "config": {
                                    "role": "Code Reviewer",
                                    "system_prompt": "Review code for security issues"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-3",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Risk Analyst"},
                                    "details": {"text": "Risk assessment"}
                                },
                                "config": {
                                    "role": "Risk Analyst",
                                    "system_prompt": "Assess and quantify security risks"
                                }
                            },
                            {
                                "type": "custom.AgentNode",
                                "id": "agent-4",
                                "templateId": "langchain-agent",
                                "attrs": {
                                    "title": {"text": "Report Generator"},
                                    "details": {"text": "Consolidate findings"}
                                },
                                "config": {
                                    "role": "Report Generator",
                                    "system_prompt": "Create security assessment report"
                                }
                            },
                            {
                                "type": "custom.CircleNode",
                                "id": "end-1",
                                "templateId": "end-sink",
                                "attrs": {
                                    "title": {"text": "End"},
                                    "details": {"text": "Security report"}
                                }
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "start-1", "port": "start-1-out"},
                                "target": {"id": "agent-1", "port": "agent-1-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "start-1", "port": "start-1-out"},
                                "target": {"id": "agent-2", "port": "agent-2-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "start-1", "port": "start-1-out"},
                                "target": {"id": "agent-3", "port": "agent-3-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-1", "port": "agent-1-out"},
                                "target": {"id": "agent-4", "port": "agent-4-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-2", "port": "agent-2-out"},
                                "target": {"id": "agent-4", "port": "agent-4-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-3", "port": "agent-3-out"},
                                "target": {"id": "agent-4", "port": "agent-4-in"}
                            },
                            {
                                "type": "standard.Link",
                                "source": {"id": "agent-4", "port": "agent-4-out"},
                                "target": {"id": "end-1", "port": "end-1-in"}
                            }
                        ]
                    }
                }
            }
        ]
    
    @staticmethod
    def get_template_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get a specific template by name"""
        templates = WorkflowTemplates.get_all_templates()
        for template in templates:
            if template["name"] == name:
                return template
        return None
    
    @staticmethod
    def get_templates_by_category(category: str) -> List[Dict[str, Any]]:
        """Get templates by category"""
        templates = WorkflowTemplates.get_all_templates()
        return [t for t in templates if t.get("category") == category]
    
    @staticmethod
    def get_categories() -> List[str]:
        """Get all available categories"""
        templates = WorkflowTemplates.get_all_templates()
        categories = set(t.get("category", "Other") for t in templates)
        return sorted(list(categories))

