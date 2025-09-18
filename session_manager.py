#!/usr/bin/env python3
"""
Session management for uroflow analysis
Uses filesystem as single source of truth for session state
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional, Dict, List

class SessionManager:
    """Manages uroflow test sessions with filesystem-based state"""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize session manager with base directory"""
        if base_dir is None:
            self.base_dir = Path.home() / '.uroflow'
        else:
            self.base_dir = Path(base_dir)

        self.sessions_dir = self.base_dir / 'sessions'
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.latest_link = self.sessions_dir / 'latest'

    def create_session(self, patient_name: Optional[str] = None, video_path: Optional[Path] = None) -> Path:
        """Create a new session directory with timestamp and optional patient name"""
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

        if patient_name:
            # Sanitize patient name for filesystem
            safe_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            session_name = f"{timestamp}-{safe_name}"
        else:
            session_name = timestamp

        session_path = self.sessions_dir / session_name
        session_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (session_path / 'frames').mkdir(exist_ok=True)

        # Save minimal metadata - only what can't be deduced from filesystem
        if video_path or patient_name:
            metadata = {
                'created': timestamp,
                'patient_name': patient_name,
                'video_path': str(video_path) if video_path else None,
                'video_hash': self._calculate_video_hash(video_path) if video_path else None
            }
            with open(session_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        # Update latest symlink
        if self.latest_link.exists() or self.latest_link.is_symlink():
            self.latest_link.unlink()
        self.latest_link.symlink_to(session_path.name)

        return session_path

    def get_session(self, session_id: Optional[str] = None) -> Optional[Path]:
        """Get session path by ID or return latest session"""
        if session_id is None or session_id == 'latest':
            if self.latest_link.exists():
                return self.latest_link.resolve()
            else:
                # Find most recent session by directory name (timestamp)
                sessions = self.list_sessions()
                if sessions:
                    return Path(sessions[0]['path'])
                return None

        # Try exact match first
        session_path = self.sessions_dir / session_id
        if session_path.exists():
            return session_path

        # Try partial match
        for path in self.sessions_dir.iterdir():
            if path.is_dir() and session_id in path.name:
                return path

        return None

    def list_sessions(self) -> List[Dict]:
        """List all sessions with their filesystem-based status"""
        sessions = []

        for session_path in self.sessions_dir.iterdir():
            if session_path.is_dir() and session_path.name != 'latest':
                # Deduce status from filesystem
                session_info = {
                    'id': session_path.name,
                    'path': str(session_path),
                    'created': self._extract_timestamp_from_name(session_path.name),
                    'patient_name': self._extract_patient_from_name(session_path.name),
                    'steps': {
                        'frames_extracted': self._has_frames(session_path),
                        'ocr_completed': self._has_ocr_data(session_path),
                        'analysis_completed': self._has_chart(session_path),
                        'report_generated': self._has_report(session_path)
                    }
                }

                # Load additional metadata if exists
                metadata_file = session_path / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            session_info['patient_name'] = metadata.get('patient_name') or session_info['patient_name']
                    except:
                        pass

                sessions.append(session_info)

        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['created'], reverse=True)
        return sessions

    def get_or_create_session_from_video(self, video_path: Path, patient_name: Optional[str] = None) -> Path:
        """Get existing session for video or create new one"""
        video_hash = self._calculate_video_hash(video_path)

        # Check if we already have a session for this video
        for session_path in self.sessions_dir.iterdir():
            if session_path.is_dir() and session_path.name != 'latest':
                metadata_file = session_path / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if metadata.get('video_hash') == video_hash:
                                # Found existing session for this video
                                return session_path
                    except:
                        continue

        # Create new session
        return self.create_session(patient_name, video_path)

    def should_extract_frames(self, session_path: Path, video_path: Path = None) -> bool:
        """Check if frame extraction is needed based on filesystem state"""
        frames_dir = session_path / 'frames'

        # Check if frames exist
        if not frames_dir.exists():
            return True

        frame_files = list(frames_dir.glob('frame_*.jpg'))
        if not frame_files:
            return True

        # If video provided, check if it matches the session's video
        if video_path:
            metadata_file = session_path / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        stored_hash = metadata.get('video_hash')
                        current_hash = self._calculate_video_hash(video_path)
                        if stored_hash and stored_hash != current_hash:
                            return True  # Video has changed
                except:
                    pass

        return False

    def should_run_ocr(self, session_path: Path) -> bool:
        """Check if OCR processing is needed based on filesystem state"""
        csv_path = session_path / 'weight_data.csv'
        json_path = session_path / 'weight_data.json'

        # Check if both OCR outputs exist and have content
        if csv_path.exists() and json_path.exists():
            if csv_path.stat().st_size > 0 and json_path.stat().st_size > 0:
                return False

        return True

    def get_session_status(self, session_path: Path) -> Dict:
        """Get current status of session based on filesystem"""
        return {
            'frames_extracted': self._has_frames(session_path),
            'ocr_completed': self._has_ocr_data(session_path),
            'analysis_completed': self._has_chart(session_path),
            'report_generated': self._has_report(session_path)
        }

    # Helper methods to check filesystem state
    def _has_frames(self, session_path: Path) -> bool:
        """Check if session has extracted frames"""
        frames_dir = session_path / 'frames'
        if frames_dir.exists():
            return len(list(frames_dir.glob('frame_*.jpg'))) > 0
        return False

    def _has_ocr_data(self, session_path: Path) -> bool:
        """Check if session has OCR data"""
        csv_path = session_path / 'weight_data.csv'
        json_path = session_path / 'weight_data.json'
        return (csv_path.exists() and csv_path.stat().st_size > 0 and
                json_path.exists() and json_path.stat().st_size > 0)

    def _has_chart(self, session_path: Path) -> bool:
        """Check if session has generated chart"""
        return (session_path / 'uroflow_chart.png').exists()

    def _has_report(self, session_path: Path) -> bool:
        """Check if session has PDF report"""
        return (session_path / 'report.pdf').exists()

    def _extract_timestamp_from_name(self, session_name: str) -> str:
        """Extract timestamp from session directory name"""
        # Format: YYYY-MM-DD-HHMMSS or YYYY-MM-DD-HHMMSS-patient
        parts = session_name.split('-')
        if len(parts) >= 4:
            return f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[3]}"
        return session_name

    def _extract_patient_from_name(self, session_name: str) -> Optional[str]:
        """Extract patient name from session directory name"""
        # Format: YYYY-MM-DD-HHMMSS-patient_name
        parts = session_name.split('-', 4)
        if len(parts) > 4:
            return parts[4].replace('_', ' ')
        return None

    def _calculate_video_hash(self, video_path: Optional[Path]) -> Optional[str]:
        """Calculate hash of video file for caching"""
        if not video_path or not video_path.exists():
            return None

        # Use file size and modification time for quick hash
        # For large video files, reading entire content would be slow
        stat = video_path.stat()
        hash_input = f"{video_path.name}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def export_session(self, session_path: Path, output_dir: Path):
        """Export session data to specified directory"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy important files
        files_to_export = [
            'weight_data.csv',
            'weight_data.json',
            'uroflow_chart.png',
            'report.pdf',
            'metadata.json'
        ]

        for filename in files_to_export:
            src = session_path / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)