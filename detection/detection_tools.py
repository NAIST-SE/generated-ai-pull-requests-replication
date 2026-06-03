"""
Detection Tools Module

This module provides detector implementations for AI-generated content detection.
All detectors inherit from BaseDetector and return standardized output format.

Available Detectors:
- ZeroGPTDetector: API-based detection using ZeroGPT service
- GPTZeroDetector: API-based detection using GPTZero (Official) service
- RoBERTaDetector: Local model-based detection using RoBERTa (chatgpt-detector-roberta)
- RoBERTaQADetector: Local model-based detection using RoBERTa-QA (chatgpt-qa-detector-roberta)

Each detector returns a standardized format:
{
    "detector": str,              # Detector name
    "ai_probability": float,      # 0.0 - 1.0
    "is_ai_generated": bool,      # True if detected as AI
    "raw_response": str,          # Full JSON response
    "success": bool,              # Whether detection succeeded
    "error": str | None,          # Error message if failed
    "model_version": str          # Model/API version
}
"""

import itertools
import json
import random
import time
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Dict, Iterator, List, Optional, Tuple

import requests

# Try to import torch and transformers (for RoBERTa)
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    print("Warning: torch/transformers not available. RoBERTa detector will not work.")


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.

    All detectors must implement:
    - detect(text): Run detection and return standardized format
    - get_name(): Return detector name (lowercase, no spaces)
    - get_model_version(): Return model/API version for reproducibility
    """

    @abstractmethod
    def detect(self, text: str) -> Dict:
        """
        Run detection on text and return standardized format.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with standardized detection results:
            {
                "detector": str,
                "ai_probability": float,
                "is_ai_generated": bool,
                "raw_response": str,
                "success": bool,
                "error": str | None,
                "model_version": str
            }
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get detector name (lowercase, no spaces).

        Returns:
            Detector name (e.g., 'zerogpt', 'roberta')
        """
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """
        Get model/API version for reproducibility tracking.

        Returns:
            Version string (e.g., 'zerogpt-api-v2', 'Hello-SimpleAI/chatgpt-detector-roberta')
        """
        pass

    def is_text_significant(self, text: str, min_chars: int = 250) -> bool:
        """
        Check if text is significant enough for detection.

        Args:
            text: Text to check
            min_chars: Minimum character count

        Returns:
            True if text is significant
        """
        return bool(text and isinstance(text, str) and len(text.strip()) >= min_chars)


class ZeroGPTDetector(BaseDetector):
    """
    ZeroGPT API-based detector.

    Uses rotating API tokens for rate limit management.
    Features:
    - Token rotation for load balancing
    - Automatic retry logic
    - Rate limit handling
    """

    API_URL = "https://api.zerogpt.com/api/detect/detectText"
    API_VERSION = "zerogpt-api-v2"

    def __init__(
        self,
        tokens_file: str,
        ignore_min_length: bool = True,
        delay_seconds: float = 1.0,
    ):
        """
        Initialize ZeroGPT detector.

        Args:
            tokens_file: Path to file with API tokens (one per line)
            ignore_min_length: Whether to ignore minimum text length check
            delay_seconds: Time to wait before each request (to avoid rate limits)
        """
        self.tokens_file = tokens_file
        self.ignore_min_length = ignore_min_length
        self.delay_seconds = delay_seconds
        self.tokens, self.token_iterator = self._load_tokens()

    def _load_tokens(self) -> Tuple[List[str], Iterator[str]]:
        """Load and rotate ZeroGPT API tokens for round-robin usage."""
        with open(self.tokens_file, "r") as file:
            tokens = file.read().splitlines()

        # Start from a random position for better distribution
        start_index = random.randint(0, len(tokens) - 1)
        rotated_tokens = tokens[start_index:] + tokens[:start_index]

        return tokens, itertools.cycle(rotated_tokens)

    def get_name(self) -> str:
        """Get detector name."""
        return "zerogpt"

    def get_model_version(self) -> str:
        """Get API version."""
        return self.API_VERSION

    def detect(self, text: str) -> Dict:
        """
        Run ZeroGPT detection on text.

        Args:
            text: Text to analyze

        Returns:
            Standardized detection result dictionary
        """
        # Check text significance (unless ignored)
        if not self.ignore_min_length and not self.is_text_significant(text):
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": "Text too short"}),
                "success": False,
                "error": "Text too short or empty",
                "model_version": self.get_model_version(),
            }

        # Get next token
        current_token = next(self.token_iterator)

        # Rate limiting delay
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        # Prepare API request
        payload = json.dumps({"input_text": text})
        headers = {
            "ApiKey": current_token,
            "Content-Type": "application/json",
        }

        try:
            # Make API request
            response = requests.post(self.API_URL, headers=headers, data=payload)
            response_data = response.json()

            # Check response code
            code = response_data.get("code")
            if code != HTTPStatus.OK:
                return {
                    "detector": self.get_name(),
                    "ai_probability": None,
                    "is_ai_generated": None,
                    "raw_response": json.dumps(response_data),
                    "success": False,
                    "error": f"API error: code {code}",
                    "model_version": self.get_model_version(),
                }

            # Extract detection results
            data = response_data.get("data", {})
            ai_probability = (
                data.get("fakePercentage", 0) / 100.0
            )  # Convert to 0-1 range
            is_ai_generated = not data.get("isHuman", True)  # isHuman=False means AI

            return {
                "detector": self.get_name(),
                "ai_probability": ai_probability,
                "is_ai_generated": is_ai_generated,
                "raw_response": json.dumps(response_data),
                "success": True,
                "error": None,
                "model_version": self.get_model_version(),
            }

        except Exception as e:
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": str(e)}),
                "success": False,
                "error": str(e),
                "model_version": self.get_model_version(),
            }


class GPTZeroDetector(BaseDetector):
    """
    GPTZero API-based detector (Official API).

    Documentation: https://gptzero.stoplight.io/docs/gptzero-api/
    """

    API_URL = "https://api.gptzero.me/v2/predict/text"
    # Using specific model version from user request
    MODEL_VERSION = "2025-12-18-base"

    def __init__(self, api_key: str):
        """
        Initialize GPTZero detector.

        Args:
            api_key: GPTZero API key string
        """
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("GPTZero API Key cannot be empty")

    def get_name(self) -> str:
        return "gptzero"

    def get_model_version(self) -> str:
        return self.MODEL_VERSION

    def detect(self, text: str) -> Dict:
        """
        Run GPTZero detection on text.
        """
        if not self.is_text_significant(text):
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": "Text too short"}),
                "success": False,
                "error": "Text too short or empty",
                "model_version": self.get_model_version(),
            }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        data = {
            "document": text,
            "multilingual": False,
            "modelVersion": self.MODEL_VERSION,
        }

        try:
            response = requests.post(self.API_URL, headers=headers, json=data)

            # Allow for some non-200 successes if API structure allows, but usually <300 is good
            if response.status_code >= 400:
                return {
                    "detector": self.get_name(),
                    "ai_probability": None,
                    "is_ai_generated": None,
                    "raw_response": response.text,
                    "success": False,
                    "error": f"API Status {response.status_code}: {response.text}",
                    "model_version": self.get_model_version(),
                }

            resp_json = response.json()

            # Parse Response
            # The structure is: { documents: [ { class_probabilities: { ai: ... }, completely_generated_prob: ... } ] }
            documents = resp_json.get("documents", [])
            if not documents:
                return {
                    "detector": self.get_name(),
                    "ai_probability": None,
                    "is_ai_generated": None,
                    "raw_response": json.dumps(resp_json),
                    "success": False,
                    "error": "No documents returned in response",
                    "model_version": self.get_model_version(),
                }

            doc = documents[0]

            # Extract detailed metrics
            class_probs = doc.get("class_probabilities", {})
            ai_prob = class_probs.get("ai", 0)

            # Additional metadata as requested
            # confidence_category = doc.get("confidence_category", "unknown")
            # confidence_score = doc.get("confidence_score", 0)
            predicted_class = doc.get("predicted_class", "unknown")

            # Construct a comprehensive result object to store in raw_response (or separate columns if extended later)
            # For now, we utilize the standardized dict but ensure 'raw_response' contains everything needed for deep analysis

            final_prob = ai_prob

            # Determine boolean label
            is_ai = predicted_class == "ai" or predicted_class == "mixed"

            # Augment raw response to ensure we capture the threshold details if they aren't fully in the original JSON (they likely are)
            # The API response already includes 'confidence_thresholds_raw', so just dumping the full JSON is sufficient to "keep" them.
            # However, prompt asked to "keep" them, implying they might be needed in the struct.
            # Since BaseDetector enforces a strict return schema, we can't easily add top-level keys without breaking polymorphism.
            # Best approach: Ensure the full response is designated as the source of truth for these metrics.

            return {
                "detector": self.get_name(),
                "ai_probability": final_prob,
                "is_ai_generated": is_ai,
                "raw_response": json.dumps(
                    resp_json
                ),  # Contains confidence_thresholds_raw, confidence_category, etc.
                "success": True,
                "error": None,
                "model_version": self.get_model_version(),
            }

        except Exception as e:
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": str(e)}),
                "success": False,
                "error": str(e),
                "model_version": self.get_model_version(),
            }


class RoBERTaDetector(BaseDetector):
    """
    RoBERTa-based local model detector.

    Uses the Hello-SimpleAI/chatgpt-detector-roberta model for detection.
    Features:
    - Local inference (no API calls)
    - GPU support if available
    - Batch processing capability
    """

    DEFAULT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize RoBERTa detector.

        Args:
            model_name: HuggingFace model name (default: Hello-SimpleAI/chatgpt-detector-roberta)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if (
            not TORCH_AVAILABLE
            or AutoTokenizer is None
            or AutoModelForSequenceClassification is None
            or torch is None
        ):
            raise ImportError(
                "torch and transformers are required for RoBERTa detector. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading RoBERTa model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("RoBERTa model loaded successfully!")

    def get_name(self) -> str:
        """Get detector name."""
        return "roberta"

    def get_model_version(self) -> str:
        """Get model name/version."""
        return self.model_name

    def detect(self, text: str) -> Dict:
        """
        Run RoBERTa detection on text.

        Args:
            text: Text to analyze

        Returns:
            Standardized detection result dictionary
        """
        # Runtime check for torch availability (for type checker)
        if torch is None:
            raise RuntimeError("torch is not available")

        # Check text significance
        if not self.is_text_significant(text):
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": "Text too short"}),
                "success": False,
                "error": "Text too short or empty",
                "model_version": self.get_model_version(),
            }

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,  # RoBERTa max length
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Extract probabilities
            # Model outputs: [human_prob, ai_prob]
            prob_human = probs[0][0].item()
            prob_ai = probs[0][1].item()

            # Determine if AI-generated (threshold at 0.5)
            is_ai_generated = prob_ai > prob_human

            # Build response
            response_data = {
                "human_probability": prob_human,
                "ai_probability": prob_ai,
                "model": self.model_name,
                "device": self.device,
            }

            return {
                "detector": self.get_name(),
                "ai_probability": prob_ai,
                "is_ai_generated": is_ai_generated,
                "raw_response": json.dumps(response_data),
                "success": True,
                "error": None,
                "model_version": self.get_model_version(),
            }

        except Exception as e:
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": str(e)}),
                "success": False,
                "error": str(e),
                "model_version": self.get_model_version(),
            }


class RoBERTaQADetector(BaseDetector):
    """
    RoBERTa-QA-based local model detector.

    Uses the Hello-SimpleAI/chatgpt-qa-detector-roberta model.
    """

    DEFAULT_MODEL = "Hello-SimpleAI/chatgpt-qa-detector-roberta"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize RoBERTa-QA detector.

        Args:
            model_name: HuggingFace model name (default: Hello-SimpleAI/chatgpt-qa-detector-roberta)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if (
            not TORCH_AVAILABLE
            or AutoTokenizer is None
            or AutoModelForSequenceClassification is None
            or torch is None
        ):
            raise ImportError(
                "torch and transformers are required for RoBERTa-QA detector. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading RoBERTa-QA model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("RoBERTa-QA model loaded successfully!")

    def get_name(self) -> str:
        """Get detector name."""
        return "roberta-qa"

    def get_model_version(self) -> str:
        """Get model name/version."""
        return self.model_name

    def detect(self, text: str) -> Dict:
        """
        Run RoBERTa-QA detection on text.

        Args:
            text: Text to analyze

        Returns:
            Standardized detection result dictionary
        """
        # Runtime check for torch availability (for type checker)
        if torch is None:
            raise RuntimeError("torch is not available")

        # Check text significance
        if not self.is_text_significant(text):
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": "Text too short"}),
                "success": False,
                "error": "Text too short or empty",
                "model_version": self.get_model_version(),
            }

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,  # RoBERTa max length
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Extract probabilities
            # Model outputs: [human_prob, ai_prob]
            prob_human = probs[0][0].item()
            prob_ai = probs[0][1].item()

            # Determine if AI-generated (threshold at 0.5)
            is_ai_generated = prob_ai > prob_human

            # Build response
            response_data = {
                "human_probability": prob_human,
                "ai_probability": prob_ai,
                "model": self.model_name,
                "device": self.device,
            }

            return {
                "detector": self.get_name(),
                "ai_probability": prob_ai,
                "is_ai_generated": is_ai_generated,
                "raw_response": json.dumps(response_data),
                "success": True,
                "error": None,
                "model_version": self.get_model_version(),
            }

        except Exception as e:
            return {
                "detector": self.get_name(),
                "ai_probability": None,
                "is_ai_generated": None,
                "raw_response": json.dumps({"error": str(e)}),
                "success": False,
                "error": str(e),
                "model_version": self.get_model_version(),
            }


# Registry of available detectors
AVAILABLE_DETECTORS = {
    "zerogpt": ZeroGPTDetector,
    "gptzero": GPTZeroDetector,
    "roberta": RoBERTaDetector,
    "roberta-qa": RoBERTaQADetector,
}


def get_available_detectors() -> List[str]:
    """
    Get list of available detector names.

    Returns:
        List of detector names
    """
    available = ["zerogpt", "gptzero"]  # API-based detectors are always available

    if TORCH_AVAILABLE:
        available.append("roberta")
        available.append("roberta-qa")

    return available


def create_detector(detector_name: str, **kwargs) -> BaseDetector:
    """
    Factory function to create a detector instance.

    Args:
        detector_name: Name of the detector ('zerogpt', 'roberta')
        **kwargs: Detector-specific arguments

    Returns:
        Detector instance

    Raises:
        ValueError: If detector name is not recognized
    """
    detector_name = detector_name.lower()

    if detector_name not in AVAILABLE_DETECTORS:
        raise ValueError(
            f"Unknown detector: {detector_name}. "
            f"Available detectors: {get_available_detectors()}"
        )

    detector_class = AVAILABLE_DETECTORS[detector_name]

    # Check if detector is available
    if detector_name == "roberta" and not TORCH_AVAILABLE:
        raise ImportError(
            "RoBERTa detector requires torch and transformers. "
            "Install with: pip install torch transformers"
        )

    return detector_class(**kwargs)
