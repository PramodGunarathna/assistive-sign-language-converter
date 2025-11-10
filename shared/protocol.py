"""
Shared protocol definitions for doctor-patient communication system.
This module contains message types, constants, and utility functions.
"""

import json
import datetime
from enum import Enum
from typing import Dict, Any

class MessageType(Enum):
    """Message types for communication protocol."""
    PATIENT_INFO = "patient_info"
    DOCTOR_MESSAGE = "doctor_message"
    PATIENT_MESSAGE = "patient_message"
    DOCTOR_VOICE = "doctor_voice"
    SYSTEM = "system"
    CONNECTION_ESTABLISHED = "connection_established"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"

class ConnectionStatus(Enum):
    """Connection status constants."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"

# Network configuration
DEFAULT_PORT = 12345
BUFFER_SIZE = 1024
MAX_CONNECTIONS = 10

class MessageProtocol:
    """Utility class for handling message protocol operations."""
    
    @staticmethod
    def create_message(message_type: MessageType, content: str, 
                      sender: str = None, recipient: str = None, 
                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a standardized message object.
        
        Args:
            message_type: Type of message
            content: Message content
            sender: Sender identifier
            recipient: Recipient identifier (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dictionary representing the message
        """
        message = {
            'type': message_type.value,
            'message': content,
            'timestamp': datetime.datetime.now().isoformat(),
            'sender': sender,
        }
        
        if recipient:
            message['recipient'] = recipient
            
        if metadata:
            message['metadata'] = metadata
            
        return message
    
    @staticmethod
    def serialize_message(message: Dict[str, Any]) -> bytes:
        """
        Serialize message to bytes for network transmission.
        
        Args:
            message: Message dictionary
            
        Returns:
            Serialized message as bytes
        """
        return json.dumps(message).encode('utf-8')
    
    @staticmethod
    def deserialize_message(data: bytes) -> Dict[str, Any]:
        """
        Deserialize bytes to message dictionary.
        
        Args:
            data: Serialized message bytes
            
        Returns:
            Message dictionary
        """
        return json.loads(data.decode('utf-8'))
    
    @staticmethod
    def create_patient_info_message(name: str, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a patient info message for initial connection.
        
        Args:
            name: Patient name
            additional_info: Additional patient information
            
        Returns:
            Patient info message
        """
        metadata = {'name': name}
        if additional_info:
            metadata.update(additional_info)
            
        return MessageProtocol.create_message(
            MessageType.PATIENT_INFO,
            f"Patient {name} connecting",
            sender=name,
            metadata=metadata
        )
    
    @staticmethod
    def create_system_message(content: str) -> Dict[str, Any]:
        """
        Create a system message.
        
        Args:
            content: System message content
            
        Returns:
            System message
        """
        return MessageProtocol.create_message(
            MessageType.SYSTEM,
            content,
            sender="SYSTEM"
        )
    
    @staticmethod
    def create_heartbeat_message() -> Dict[str, Any]:
        """
        Create a heartbeat message to keep connection alive.
        
        Returns:
            Heartbeat message
        """
        return MessageProtocol.create_message(
            MessageType.HEARTBEAT,
            "heartbeat",
            sender="heartbeat"
        )
    
    @staticmethod
    def format_timestamp(timestamp_str: str = None) -> str:
        """
        Format timestamp for display.
        
        Args:
            timestamp_str: ISO timestamp string (optional)
            
        Returns:
            Formatted timestamp string
        """
        if timestamp_str:
            dt = datetime.datetime.fromisoformat(timestamp_str)
        else:
            dt = datetime.datetime.now()
            
        return dt.strftime("%H:%M:%S")
    
    @staticmethod
    def format_display_message(message_data: Dict[str, Any]) -> str:
        """
        Format message for display in chat interface.
        
        Args:
            message_data: Message dictionary
            
        Returns:
            Formatted display string
        """
        timestamp = MessageProtocol.format_timestamp(message_data.get('timestamp'))
        sender = message_data.get('sender', 'Unknown')
        message = message_data.get('message', '')
        message_type = message_data.get('type', '')
        
        if message_type == MessageType.SYSTEM.value:
            return f"[{timestamp}] SYSTEM: {message}"
        else:
            return f"[{timestamp}] {sender}: {message}"

class NetworkUtils:
    """Utility functions for network operations."""
    
    @staticmethod
    def get_local_ip() -> str:
        """
        Get the local IP address of the machine.
        
        Returns:
            Local IP address string
        """
        import socket
        try:
            # Create a temporary socket to get local IP
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_socket.connect(("8.8.8.8", 80))
            local_ip = temp_socket.getsockname()[0]
            temp_socket.close()
            return local_ip
        except:
            return "127.0.0.1"
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """
        Validate if the given string is a valid IP address.
        
        Args:
            ip: IP address string to validate
            
        Returns:
            True if valid IP, False otherwise
        """
        import socket
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    @staticmethod
    def is_port_available(port: int, host: str = 'localhost') -> bool:
        """
        Check if a port is available for binding.
        
        Args:
            port: Port number to check
            host: Host address (default: localhost)
            
        Returns:
            True if port is available, False otherwise
        """
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return True
        except socket.error:
            return False
