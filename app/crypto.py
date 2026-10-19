"""
Encrypted Storage at Rest for Patient Fundus Photographs.
Uses AES-GCM (Fernet) symmetric encryption to ensure patient health information (PHI)
and raw uploaded retinal images are never stored in plaintext on disk.
"""

import os
from pathlib import Path
from typing import Union, Tuple
from cryptography.fernet import Fernet


class EncryptedStorage:
    """
    Encrypts files before writing to disk and decrypts them in-memory upon authorized retrieval.
    """

    def __init__(self, storage_dir: str = "app/uploads_encrypted", key_path: str = "app/storage_key.key"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.key_path = Path(key_path)

        # Load or generate symmetric key
        if self.key_path.exists():
            with open(self.key_path, "rb") as f:
                self.key = f.read().strip()
        else:
            self.key = Fernet.generate_key()
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.key_path, "wb") as f:
                f.write(self.key)

        self.cipher = Fernet(self.key)

    def save_encrypted(self, raw_bytes: bytes, filename: str) -> str:
        """
        Encrypts raw bytes and writes ciphertext to disk.
        Returns stored filename.
        """
        encrypted_data = self.cipher.encrypt(raw_bytes)
        enc_filename = f"{filename}.enc"
        dest_path = self.storage_dir / enc_filename

        with open(dest_path, "wb") as f:
            f.write(encrypted_data)

        return enc_filename

    def read_decrypted(self, enc_filename: str) -> bytes:
        """
        Reads ciphertext from disk and returns decrypted raw bytes in-memory.
        """
        dest_path = self.storage_dir / enc_filename
        if not dest_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {dest_path}")

        with open(dest_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        return decrypted_bytes
