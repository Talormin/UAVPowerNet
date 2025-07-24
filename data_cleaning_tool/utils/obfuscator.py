import hashlib
import numpy as np
from typing import Any


class DataObfuscator:
    """Data transformation utilities for privacy preservation"""

    @staticmethod
    def _generate_salt(key: str, length: int = 8) -> bytes:
        """Create deterministic salt value"""
        return hashlib.blake2b(key.encode(), digest_size=length).digest()

    @staticmethod
    def transform_identifiers(data: Any,
                              key: str = 'default') -> Any:
        """
        Apply consistent one-way transformation to sensitive values

        Args:
            data: Value or array to transform
            key: Obfuscation seed value

        Returns:
            Transformed data
        """
        salt = DataObfuscator._generate_salt(key)
        if isinstance(data, (list, np.ndarray)):
            return np.array([hashlib.blake2b(str(x).encode(),
                                             salt=salt).hexdigest()[:16] for x in data])
        else:
            return hashlib.blake2b(str(data).encode(),
                                   salt=salt).hexdigest()[:16]