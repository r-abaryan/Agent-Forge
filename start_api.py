"""
Start AgentForge API Server
Convenience script to launch the REST API
"""

import uvicorn
import argparse


def main():
    parser = argparse.ArgumentParser(description="Start AgentForge API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          AgentForge REST API Server                      ║
╠══════════════════════════════════════════════════════════╣
║  Server: http://{args.host}:{args.port}                  ║
║  Docs:   http://{args.host}:{args.port}/docs             ║
║  Health: http://{args.host}:{args.port}/health           ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
