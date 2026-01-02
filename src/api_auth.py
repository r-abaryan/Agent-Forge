"""
API Authentication - API key management and validation
"""

import secrets
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """Manages API keys for authentication"""
    
    def __init__(self, keys_file: str = "api_keys.json"):
        """
        Initialize API key manager.
        
        Args:
            keys_file: Path to JSON file storing API keys
        """
        self.keys_file = Path(keys_file)
        self.keys: Dict[str, Dict] = {}
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from file"""
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r') as f:
                    self.keys = json.load(f)
            except Exception as e:
                print(f"Error loading API keys: {e}")
                self.keys = {}
        else:
            # Create default admin key
            admin_key = self.generate_key("admin", "Administrator")
            print(f"\n{'='*60}")
            print(f"🔑 ADMIN API KEY GENERATED")
            print(f"{'='*60}")
            print(f"Key: {admin_key}")
            print(f"Save this key securely - it won't be shown again!")
            print(f"{'='*60}\n")
    
    def save_keys(self):
        """Save API keys to file"""
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            print(f"Error saving API keys: {e}")
    
    def generate_key(self, name: str, description: str = "") -> str:
        """
        Generate a new API key.
        
        Args:
            name: Name/identifier for the key
            description: Description of the key's purpose
            
        Returns:
            Generated API key
        """
        api_key = f"agf_{secrets.token_urlsafe(32)}"
        
        self.keys[api_key] = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "request_count": 0
        }
        
        self.save_keys()
        return api_key
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if api_key in self.keys:
            # Update last used timestamp and count
            self.keys[api_key]["last_used"] = datetime.now().isoformat()
            self.keys[api_key]["request_count"] += 1
            self.save_keys()
            return True
        return False
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if api_key in self.keys:
            del self.keys[api_key]
            self.save_keys()
            return True
        return False
    
    def list_keys(self) -> List[Dict]:
        """
        List all API keys (without showing the actual keys).
        
        Returns:
            List of key information
        """
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "created_at": info["created_at"],
                "last_used": info["last_used"],
                "request_count": info["request_count"]
            }
            for info in self.keys.values()
        ]


# Global API key manager instance
api_key_manager = APIKeyManager()


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Dependency for verifying API keys.
    
    Args:
        api_key: API key from request header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required. Include 'X-API-Key' header."
        )
    
    if not api_key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return api_key
