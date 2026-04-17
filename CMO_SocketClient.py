# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:58:07 2025

@author: Command Dev Team
"""

import socket
import json
import re


class CMO_SocketClient:
    """
    A simple, self-contained socket client class to send and receive data from Command PE.
    All logic is handled internally.
    """
    def __init__(self, host, port, timeout=5, stop_word='\r\n\r\n'):
        """
        Initializes the client.

        Args:
            host (str): The server IP address.
            port (int): The server port.
            timeout (int): Socket timeout in seconds.
            stop_word (str): The delimiter that marks the end of a message.
        
        Note:
            The stop_word is used to determine message boundaries. Make sure it matches
            the server's message termination protocol.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.stop_word = stop_word
        self.socket = None

    def connect(self):
        """Establishes connection to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            print(f"Successfully connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect. Error: {e}")
            self.socket = None
            raise # Re-raise the exception to be handled by the caller

    def disconnect(self):
        """Closes the connection."""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Connection closed.")

    def send(self, message: str):
        """
        Encodes a string message and sends it to the server.

        Args:
            message (str): The command or message to send.
        """
        if not self.socket:
            raise ConnectionError("Cannot send: not connected.")
        try:
            self.socket.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Error sending message: {e}")
            raise

    def receive(self, format='string'):
        """
        Receives data and processes it based on the specified format.

        Args:
            format (str): The desired output format.
                          Options: 'string', 'json', 'lua_table', 'auto'.
                          'auto' attempts to convert values to proper types.
                          Defaults to 'string'.

        Returns:
            The processed data (dict or str), or None on error.
        """
        if not self.socket:
            raise ConnectionError("Cannot receive: not connected.")

        raw_data_string = self._receive_raw_data()
        if raw_data_string is None:
            return None

        if format == 'json':
            try:
                return json.loads(raw_data_string)
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON. Returning raw string instead.")
                return raw_data_string
        
        elif format == 'lua_table':
            return self._parse_lua_table(raw_data_string)
            
        elif format == 'auto':
            # Try to convert to proper types
            value = raw_data_string.strip()
            # Handle booleans
            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False
            # Handle numbers
            try:
                if '.' in value:
                    return float(value)
                return int(value)
            except ValueError:
                pass
            # If no conversion possible, return as string
            return value

        else: # Default is 'string'
            return raw_data_string

    # --- Internal (Private) Methods ---

    def _receive_raw_data(self):
        """
        Internal method to read from the socket until the stop_word is found.
        
        Returns:
            Optional[str]: The received data as a string, or None if an error occurred
            
        Note:
            This method handles partial receives and will continue reading until either:
            1. The stop_word is found
            2. The connection is closed by the peer
            3. A timeout occurs
            4. An error occurs
        """
        encoded_stop_word = self.stop_word.encode('utf-8')
        accumulated_data = bytearray()
        
        while True:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    print("Connection closed by peer.")
                    return None
            except TimeoutError:
                print("Socket receive timed out.")
                break
            except Exception as e:
                print(f"Error during receive: {e}")
                return None
            
            accumulated_data.extend(chunk)
            if encoded_stop_word in accumulated_data:
                break
        
        # Find the stop word and slice the message content from the buffer
        end_index = accumulated_data.find(encoded_stop_word)
        if end_index != -1:
            message_bytes = accumulated_data[:end_index]
        else:
            message_bytes = accumulated_data

        return message_bytes.decode('utf-8', errors='ignore')

    def _parse_lua_table(self, text: str):
        """
        Internal method to parse a Lua-style string into a dictionary.
        
        Args:
            text (str): The Lua-style table string to parse
            
        Returns:
            dict: A dictionary containing the parsed key-value pairs
            
        Note:
            Handles the following Lua value types:
            - Strings (both single and double quotes)
            - Numbers (integers and floats)
            - Booleans
            - Nil (converted to None)
            - Simple nested tables (one level deep)
        """
        def clean_value(val: str):
            """Helper function to convert string values to appropriate Python types."""
            val = val.strip()
            # Handle nil
            if val.lower() == 'nil':
                return None
            # Handle booleans
            if val.lower() == 'true':
                return True
            if val.lower() == 'false':
                return False
            # Handle numbers
            try:
                if '.' in val:
                    return float(val)
                return int(val)
            except ValueError:
                pass
            # Handle strings - remove quotes
            if (val.startswith('"') and val.endswith('"')) or \
            (val.startswith("'") and val.endswith("'")):
                return val[1:-1]
            return val

        data = {}
        # Find the first opening brace, ignoring any text before it
        table_start = text.find('{')
        if table_start == -1:
            return data
        
        # Extract everything between the outermost braces
        brace_count = 0
        table_end = -1
        for i in range(table_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    table_end = i
                    break
        
        if table_end == -1:
            return data

        # Get the table contents
        table_content = text[table_start + 1:table_end]

        # Split the content into key-value pairs
        current_pair = ''
        pairs = []
        in_string = False
        string_char = None
        in_nested = 0
        
        for char in table_content:
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif char == '{' and not in_string:
                in_nested += 1
            elif char == '}' and not in_string:
                in_nested -= 1
            elif char == ',' and not in_string and in_nested == 0:
                if current_pair.strip():
                    pairs.append(current_pair.strip())
                current_pair = ''
                continue
            
            current_pair += char
        
        if current_pair.strip():
            pairs.append(current_pair.strip())

        # Process each pair
        for pair in pairs:
            # Skip empty pairs
            if not pair.strip():
                continue
                
            # Handle key-value pairs
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Handle nested tables
                if value.startswith('{'):
                    data[key] = self._parse_lua_table(value)
                else:
                    data[key] = clean_value(value)
            else:
                # Handle array-style values
                data[len(data)] = clean_value(pair)

        return data
    
    # --- Context Manager Support for 'with' statements ---
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()