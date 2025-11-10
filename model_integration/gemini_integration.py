"""
Gemini API Integration for Sign Language Translation Enhancement

This module provides integration with Google's Gemini API to enhance
sign language translation outputs by converting fragmented words into
meaningful sentences.
"""

import google.generativeai as genai
import logging
import time
import re
from typing import Optional, List, Dict

class SignLanguageLLMProcessor:
    """
    Handles post-processing of sign language recognition outputs using Gemini API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Gemini API processor.
        
        Args:
            api_key (str): Google Gemini API key
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # List available models for debugging
        try:
            models = list(genai.list_models())
            self.logger.info(f"Available models: {[model.name for model in models]}")
        except Exception as e:
            self.logger.warning(f"Could not list models: {e}")
        
        # Initialize the model (use available model names from the list)
        try:
            # Try different model names that are available (from the list we got)
            model_names = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-pro-latest', 'gemini-flash-latest']
            self.model = None
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.logger.info(f"Successfully loaded {model_name}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Failed to load any available model")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize any model: {e}")
            raise e
        
        # System prompt for sign language context
        self.system_prompt = """You are an expert assistant for processing sign language recognition outputs. Your task is to convert fragmented or repeated words from sign language recognition into coherent, meaningful sentences with EXACTLY 4 TO 5 WORDS (NO MORE, NO LESS).

Examples:
- "hello hello pain pain" → "I have severe pain"
- "doctor help help me" → "Doctor please help me"  
- "head hurt hurt bad" → "My head hurts badly"
- "water water drink" → "I need water now"
- "back pain" → "I have back pain"
- "help" → "I need help now"

CRITICAL REQUIREMENTS:
- ALWAYS generate sentences with EXACTLY 4 TO 5 WORDS (NO MORE THAN 5 WORDS)
- Use the provided words as the foundation
- Add appropriate context words to make meaningful sentences
- Remove unnecessary repetitions
- Fix grammar and sentence structure
- Keep medical terminology accurate
- Always respond in clear, simple English suitable for medical communication
- Make sentences sound natural and meaningful
- NEVER exceed 5 words

Input: {input_text}

Output:"""
        
        self.logger.info("SUCCESS: Gemini API processor initialized successfully")
    
    def process_sign_language_output(self, raw_output: str) -> str:
        """
        Process raw sign language recognition output into a meaningful sentence.
        
        Args:
            raw_output (str): Raw output from sign language recognition model
            
        Returns:
            str: Processed, meaningful sentence
        """
        try:
            if not raw_output or not raw_output.strip():
                return "No message detected"
            
            # Clean the input
            cleaned_input = self._preprocess_input(raw_output)
            
            if not cleaned_input:
                return "No clear message detected"
            
            # If input is already clean and coherent AND has 4-5 words, return as-is (truncate if needed)
            words = cleaned_input.split()
            if self._is_already_clean(cleaned_input) and 4 <= len(words) <= 5:
                return cleaned_input
            elif self._is_already_clean(cleaned_input) and len(words) > 5:
                # Truncate to 5 words if already clean but too long
                return ' '.join(words[:5])
            
            # Use Gemini API to enhance the output
            enhanced_output = self._call_gemini_api(cleaned_input)
            
            if enhanced_output:
                # Ensure the output has 4-5 words (limit to 5 words maximum)
                words = enhanced_output.split()
                word_count = len(words)
                
                if 4 <= word_count <= 5:
                    self.logger.info(f"SUCCESS: Enhanced output: '{raw_output}' -> '{enhanced_output}'")
                    return enhanced_output
                elif word_count > 5:
                    # Truncate to 5 words if too long
                    truncated = ' '.join(words[:5])
                    self.logger.info(f"SUCCESS: Truncated output (was {word_count} words): '{raw_output}' -> '{truncated}'")
                    return truncated
                else:
                    # If less than 4 words, try to enhance further
                    self.logger.warning(f"Output too short ({word_count} words), trying to enhance further")
                    further_enhanced = self._enhance_short_sentence(enhanced_output)
                    if further_enhanced:
                        further_words = further_enhanced.split()
                        if 4 <= len(further_words) <= 5:
                            return further_enhanced
                        elif len(further_words) > 5:
                            return ' '.join(further_words[:5])
                    # Fallback to basic enhancement
                    return self._basic_enhance_sentence(cleaned_input)
            else:
                # Fallback to enhanced input if API fails
                self.logger.warning(f"WARNING: API enhancement failed, using enhanced input: '{cleaned_input}'")
                return self._basic_enhance_sentence(cleaned_input)
                
        except Exception as e:
            self.logger.error(f"ERROR: Error processing sign language output: {e}")
            # Fallback to basic enhancement
            return self._basic_enhance_sentence(raw_output)
    
    def _preprocess_input(self, text: str) -> str:
        """
        Preprocess the input text to remove obvious issues.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common recognition artifacts
        text = re.sub(r'\b(unk|unknown|pad|padding)\b', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def _is_already_clean(self, text: str) -> bool:
        """
        Check if the text is already clean and doesn't need LLM processing.
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if text is already clean
        """
        if not text:
            return True
        
        # Check for obvious repetition patterns
        words = text.lower().split()
        if len(words) <= 2:
            return True
        
        # Check for excessive repetition (more than 70% of words are repeated)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_counts)
        
        # If more than 70% of words are unique, it's likely clean
        if unique_words / total_words > 0.7:
            return True
        
        # Check for obvious repetition patterns (consecutive repeated words)
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                # Found consecutive repetition, needs cleaning
                return False
        
        # Check for common clean patterns
        clean_patterns = [
            r'^(yes|no|hello|hi|thanks?|thank you)$',  # Simple responses
        ]
        
        for pattern in clean_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _call_gemini_api(self, text: str) -> Optional[str]:
        """
        Call Gemini API to enhance the text.
        
        Args:
            text (str): Input text to enhance
            
        Returns:
            Optional[str]: Enhanced text or None if failed
        """
        try:
            # Prepare the prompt
            prompt = self.system_prompt.format(input_text=text)
            
            # Generate response
            self.logger.info(f"Calling Gemini API with prompt: {prompt[:100]}...")
            
            # Try different approaches to call the API
            try:
                response = self.model.generate_content(prompt)
                self.logger.info(f"API call successful, response type: {type(response)}")
            except Exception as api_error:
                self.logger.error(f"API call failed: {api_error}")
                # Try with generation config
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=30,  # Reduced to limit output length (5 words max)
                        )
                    )
                    self.logger.info("API call with config successful")
                except Exception as config_error:
                    self.logger.error(f"API call with config also failed: {config_error}")
                    return None
            
            if response and hasattr(response, 'text') and response.text:
                enhanced_text = response.text.strip()
                self.logger.info(f"Gemini API response: '{enhanced_text}'")
                
                # Basic validation
                if len(enhanced_text) > 5:
                    return enhanced_text
                else:
                    self.logger.warning(f"API response too short: '{enhanced_text}'")
            else:
                self.logger.warning("No valid response from Gemini API")
                if response:
                    self.logger.info(f"Response object: {response}")
                    if hasattr(response, 'parts'):
                        self.logger.info(f"Response parts: {response.parts}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"ERROR: Gemini API call failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _basic_clean(self, text: str) -> str:
        """
        Basic cleaning fallback when API is not available.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Basic cleaned text
        """
        if not text:
            return "No message detected"
        
        # Remove excessive repetitions
        words = text.split()
        cleaned_words = []
        prev_word = None
        repeat_count = 0
        
        for word in words:
            if word.lower() == prev_word:
                repeat_count += 1
                if repeat_count < 2:  # Allow one repetition
                    cleaned_words.append(word)
            else:
                repeat_count = 0
                cleaned_words.append(word)
            prev_word = word.lower()
        
        cleaned_text = ' '.join(cleaned_words)
        
        # Basic sentence structure
        if cleaned_text and not cleaned_text.endswith(('.', '!', '?')):
            cleaned_text += '.'
        
        return cleaned_text.capitalize()
    
    def _enhance_short_sentence(self, sentence: str) -> Optional[str]:
        """
        Enhance a short sentence to make it 4-5 words.

        Args:
            sentence (str): Short sentence to enhance

        Returns:
            Optional[str]: Enhanced sentence (4-5 words) or None if failed
        """
        try:
            enhance_prompt = f"""Please expand this sentence to make it EXACTLY 4 TO 5 WORDS (NO MORE, NO LESS) while keeping it meaningful and natural:

Original: "{sentence}"

Requirements:
- Must be EXACTLY 4 TO 5 WORDS (NO MORE THAN 5 WORDS)
- Keep the original meaning
- Add appropriate context
- Make it sound natural
- Use medical/health context if appropriate

Enhanced sentence:"""
            
            response = self.model.generate_content(
                enhance_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=30,  # Limit to 5 words
                )
            )
            if response and response.text:
                enhanced = response.text.strip()
                words = enhanced.split()
                word_count = len(words)
                if 4 <= word_count <= 5:
                    return enhanced
                elif word_count > 5:
                    # Truncate to 5 words
                    return ' '.join(words[:5])
            return None
        except Exception as e:
            self.logger.error(f"Error enhancing short sentence: {e}")
            return None
    
    def _basic_enhance_sentence(self, words: str) -> str:
        """
        Basic enhancement to create 4-5 word sentences without API.

        Args:
            words (str): Input words

        Returns:
            str: Enhanced sentence with 4-5 words (maximum 5 words)
        """
        word_list = words.split()
        
        # Common enhancement patterns for medical/health context (4-5 words maximum)
        enhancements = {
            "pain": "I have severe pain now",
            "help": "I need help right now",
            "doctor": "I need to see doctor",
            "water": "I need water now",
            "head": "My head hurts badly",
            "back": "I have back pain",
            "hurt": "I am hurt now",
            "sick": "I am feeling sick",
            "tired": "I am very tired",
            "hungry": "I am hungry now"
        }
        
        # Check if any word matches our enhancement patterns
        for word in word_list:
            if word.lower() in enhancements:
                result = enhancements[word.lower()]
                # Ensure it's 5 words or less
                result_words = result.split()
                if len(result_words) > 5:
                    return ' '.join(result_words[:5])
                return result
        
        # Generic enhancement for any words (ensuring 4-5 words maximum)
        if len(word_list) == 1:
            return f"I need {word_list[0]} now"
        elif len(word_list) == 2:
            return f"I have {word_list[0]} {word_list[1]}"
        elif len(word_list) == 3:
            return f"I have {word_list[0]} {word_list[1]} {word_list[2]}"
        else:
            # For longer inputs, limit to first 3 words + context (max 5 words total)
            limited_words = word_list[:3]
            return f"I have {' '.join(limited_words)}"
    
    def test_connection(self) -> bool:
        """
        Test the Gemini API connection.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            test_response = self.model.generate_content("Hello, test message")
            if test_response and test_response.text:
                self.logger.info("SUCCESS: Gemini API connection test successful")
                return True
            else:
                self.logger.error("ERROR: Gemini API connection test failed - no response")
                return False
        except Exception as e:
            self.logger.error(f"ERROR: Gemini API connection test failed: {e}")
            return False


def create_llm_processor(api_key: str = "AIzaSyBn5OVixuzmfpg5oIHGjqtaVgzek0xuL2s") -> SignLanguageLLMProcessor:
    """
    Create and initialize the LLM processor.
    
    Args:
        api_key (str): Gemini API key
        
    Returns:
        SignLanguageLLMProcessor: Initialized processor instance
    """
    return SignLanguageLLMProcessor(api_key)
