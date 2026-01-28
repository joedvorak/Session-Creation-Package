"""
Title and keyword generation for sessions.

Provides LLM-based generation of session titles and keywords
using Gemini or Ollama backends.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TitleGenerationResult:
    """Result from title generation."""
    titles: List[str]
    keywords: List[str]
    model_name: str
    raw_response: Optional[str] = None


DEFAULT_PROMPT_TEMPLATE = """I am organizing oral research presentation sessions for the American Society of Agricultural and Biological Engineers Annual International Meeting. Please provide 3 options for the name/title of a session. Also provide 5 keywords describing the session. The name and keywords should highlight the commonality among all presentations. The target audience for titles and keywords is engineering designers and researchers. The title should be descriptive of the content and be interesting and engaging. It should be less than 100 characters long.

Please respond in JSON format:
{{"title_1": "Session Title", "title_2": "Session Title", "title_3": "Session Title", "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5"}}

The titles and abstracts for presentations assigned to this session are:
{presentations}"""


class TitleGenerator(ABC):
    """
    Abstract base class for title generation backends.
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
    
    @abstractmethod
    def generate(
        self,
        presentations: List[Dict[str, str]],
        prompt_template: Optional[str] = None,
    ) -> TitleGenerationResult:
        """
        Generate titles and keywords for a session.
        
        Args:
            presentations: List of dicts with 'title' and 'abstract' keys
            prompt_template: Optional custom prompt template
            
        Returns:
            TitleGenerationResult with titles and keywords
        """
        pass
    
    def _format_presentations(self, presentations: List[Dict[str, str]]) -> str:
        """Format presentations for prompt."""
        lines = []
        for i, pres in enumerate(presentations, 1):
            title = pres.get("title", "Untitled")
            abstract = pres.get("abstract", "")
            if abstract:
                lines.append(f"{i}. {title}: {abstract}")
            else:
                lines.append(f"{i}. {title}")
        return "\n\n".join(lines)
    
    def _parse_response(self, response_text: str) -> Tuple[List[str], List[str]]:
        """Parse JSON response to extract titles and keywords."""
        titles = []
        keywords = []
        
        # Try to find JSON in response
        try:
            # Handle both JSON block and inline JSON
            text = response_text.strip()
            
            # Try to find JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                
                # Extract titles
                for key in ['title_1', 'title1', 'title 1']:
                    if key in data:
                        titles.append(data[key])
                        break
                
                for key in ['title_2', 'title2', 'title 2']:
                    if key in data:
                        titles.append(data[key])
                        break
                
                for key in ['title_3', 'title3', 'title 3']:
                    if key in data:
                        titles.append(data[key])
                        break
                
                # Extract keywords
                kw_str = data.get('keywords', '')
                if isinstance(kw_str, str):
                    keywords = [k.strip() for k in kw_str.split(',')]
                elif isinstance(kw_str, list):
                    keywords = kw_str
                    
        except json.JSONDecodeError:
            # Fallback: try to extract from text
            pass
        
        return titles, keywords


class GeminiTitleGenerator(TitleGenerator):
    """
    Title generator using Google Gemini API.
    """
    
    DEFAULT_MODEL = "gemini-2.0-flash"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize Gemini title generator.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Model name for generation
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable."
            )
        
        self._model = model
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package required. Install with: pip install google-genai"
                )
        return self._client
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(
        self,
        presentations: List[Dict[str, str]],
        prompt_template: Optional[str] = None,
    ) -> TitleGenerationResult:
        """Generate titles using Gemini."""
        client = self._get_client()
        
        template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        pres_text = self._format_presentations(presentations)
        prompt = template.format(presentations=pres_text)
        
        # Try structured output first
        try:
            from google.genai import types
            from pydantic import BaseModel
            
            class SessionInfo(BaseModel):
                title_1: str
                title_2: str
                title_3: str
                keywords: str
            
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SessionInfo,
                ),
            )
            
            data = json.loads(response.text)
            titles = [data.get("title_1", ""), data.get("title_2", ""), data.get("title_3", "")]
            keywords = [k.strip() for k in data.get("keywords", "").split(",")]
            
            return TitleGenerationResult(
                titles=[t for t in titles if t],
                keywords=[k for k in keywords if k],
                model_name=self._model,
                raw_response=response.text,
            )
            
        except Exception:
            # Fallback to unstructured
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            
            titles, keywords = self._parse_response(response.text)
            
            return TitleGenerationResult(
                titles=titles,
                keywords=keywords,
                model_name=self._model,
                raw_response=response.text,
            )


class OllamaTitleGenerator(TitleGenerator):
    """
    Title generator using local Ollama server.
    """
    
    DEFAULT_MODEL = "llama3.2:3b"
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama title generator.
        
        Args:
            model: Ollama model name
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(
        self,
        presentations: List[Dict[str, str]],
        prompt_template: Optional[str] = None,
    ) -> TitleGenerationResult:
        """Generate titles using Ollama."""
        import requests
        
        template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        pres_text = self._format_presentations(presentations)
        prompt = template.format(presentations=pres_text)
        
        response = requests.post(
            f"{self._host}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=self._timeout,
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama generation failed: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        response_text = data.get("response", "")
        
        titles, keywords = self._parse_response(response_text)
        
        return TitleGenerationResult(
            titles=titles,
            keywords=keywords,
            model_name=self._model,
            raw_response=response_text,
        )


def generate_session_hash(presentation_ids: List[str]) -> str:
    """
    Generate hash for a set of presentations.
    
    Used to check if titles need regeneration when session membership changes.
    """
    sorted_ids = sorted(presentation_ids)
    return hashlib.sha256("|".join(sorted_ids).encode()).hexdigest()[:16]


def create_title_generator(
    backend: str = "gemini",
    model: Optional[str] = None,
    **kwargs
) -> TitleGenerator:
    """
    Factory function to create title generators.
    
    Args:
        backend: Backend type ('gemini' or 'ollama')
        model: Model name (uses backend default if not specified)
        **kwargs: Additional backend-specific arguments
        
    Returns:
        TitleGenerator instance
    """
    if backend == "gemini":
        return GeminiTitleGenerator(
            model=model or GeminiTitleGenerator.DEFAULT_MODEL,
            **kwargs
        )
    elif backend == "ollama":
        return OllamaTitleGenerator(
            model=model or OllamaTitleGenerator.DEFAULT_MODEL,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'gemini' or 'ollama'")


class CachedTitleGenerator:
    """
    Wrapper that caches generated titles by presentation set hash.
    
    Usage:
        generator = CachedTitleGenerator(
            GeminiTitleGenerator(),
            conference_db
        )
        result = generator.generate(session_id, presentations)
    """
    
    def __init__(self, generator: TitleGenerator, conference_db):
        """
        Initialize cached generator.
        
        Args:
            generator: Underlying title generator
            conference_db: ConferenceDB instance for caching
        """
        self.generator = generator
        self.db = conference_db
    
    def generate(
        self,
        session_id: str,
        presentations: List[Dict[str, str]],
        force_regenerate: bool = False,
        prompt_template: Optional[str] = None,
    ) -> TitleGenerationResult:
        """
        Generate titles, using cache if available.
        
        Args:
            session_id: Session ID for caching
            presentations: List of presentation dicts
            force_regenerate: Skip cache and regenerate
            prompt_template: Optional custom prompt
            
        Returns:
            TitleGenerationResult
        """
        import sqlite3
        
        # Calculate hash of current presentations
        pres_ids = [p.get("abstract_id", p.get("title", "")) for p in presentations]
        pres_hash = generate_session_hash(pres_ids)
        
        # Check cache
        if not force_regenerate:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT title_1, title_2, title_3, keywords, model_name
                    FROM generated_titles
                    WHERE session_id = ? AND presentation_set_hash = ?
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """,
                    (session_id, pres_hash)
                )
                row = cursor.fetchone()
                
                if row:
                    titles = [t for t in [row[0], row[1], row[2]] if t]
                    keywords = [k.strip() for k in row[3].split(",")] if row[3] else []
                    
                    return TitleGenerationResult(
                        titles=titles,
                        keywords=keywords,
                        model_name=row[4],
                    )
        
        # Generate new
        result = self.generator.generate(presentations, prompt_template)
        
        # Cache result
        self.db.update_session_titles(
            session_id=session_id,
            title_options=result.titles,
            keywords=result.keywords,
            model_name=result.model_name,
        )
        
        return result
