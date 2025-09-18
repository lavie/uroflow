#!/usr/bin/env python3
"""
Async OCR processor using OpenAI's native AsyncOpenAI client
Handles concurrent processing with proper rate limiting
"""
import asyncio
from asyncio import Semaphore
import aiometer
from openai import AsyncOpenAI
import base64
from pathlib import Path
from typing import List, Dict, Optional
import os
import click


class AsyncOCRProcessor:
    """Process frames concurrently using OpenAI's async client"""

    def __init__(self, api_key: str, max_concurrent: int = 10):
        """
        Initialize the async OCR processor

        Args:
            api_key: OpenAI API key
            max_concurrent: Maximum number of concurrent API calls
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def process_single_frame(self, frame_path: str) -> Dict:
        """
        Process a single frame using AsyncOpenAI

        Args:
            frame_path: Path to the frame image

        Returns:
            Dict with frame number, time, filename, weight, and error (if any)
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Read and encode image
                with open(frame_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')

                # Extract frame info
                filename = Path(frame_path).name
                frame_number = int(filename.split('_')[1].split('.')[0])
                time_seconds = (frame_number - 1) * 0.5  # 2 fps

                # Call OpenAI API (it handles retries internally!)
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Look at this image of a digital scale display. Extract ONLY the numerical weight reading shown on the display. Return just the number with its decimal point if present, nothing else. If you can't clearly see a number, return 'unclear'."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }]
                )

                weight = response.choices[0].message.content.strip()

                return {
                    'frame': frame_number,
                    'time_seconds': time_seconds,
                    'filename': frame_path,
                    'weight': weight,
                    'error': None
                }

            except Exception as e:
                # Extract frame info even on error
                filename = Path(frame_path).name
                try:
                    frame_number = int(filename.split('_')[1].split('.')[0])
                    time_seconds = (frame_number - 1) * 0.5
                except:
                    frame_number = 0
                    time_seconds = 0

                # Return error result
                return {
                    'frame': frame_number,
                    'time_seconds': time_seconds,
                    'filename': frame_path,
                    'weight': 'error',
                    'error': str(e)
                }

    async def process_all_frames_simple(self, frame_files: List[str]) -> List[Dict]:
        """
        Process all frames using asyncio.gather (simple approach)

        Args:
            frame_files: List of frame file paths

        Returns:
            List of results for each frame
        """
        tasks = [self.process_single_frame(f) for f in frame_files]
        results = await asyncio.gather(*tasks)
        await self.client.close()
        return results

    async def process_all_frames_with_rate_limit(self, frame_files: List[str],
                                                 max_per_second: float = 5) -> List[Dict]:
        """
        Process frames with explicit rate limiting using aiometer

        Args:
            frame_files: List of frame file paths
            max_per_second: Maximum API calls per second

        Returns:
            List of results for each frame
        """
        # Create tasks (need to wrap in lambda for aiometer)
        tasks = [lambda f=frame: self.process_single_frame(f) for frame in frame_files]

        # Process with rate limiting
        results = await aiometer.run_all(
            tasks,
            max_at_once=self.max_concurrent,  # Max concurrent
            max_per_second=max_per_second     # Rate limit per second
        )

        await self.client.close()
        return results

    async def process_all_frames_with_progress(self, frame_files: List[str],
                                              progress_callback=None,
                                              max_per_second: float = 5) -> List[Dict]:
        """
        Process frames with progress updates

        Args:
            frame_files: List of frame file paths
            progress_callback: Optional callback function for progress updates
            max_per_second: Maximum API calls per second

        Returns:
            List of results for each frame
        """
        results = []
        errors = []

        # Use aiometer.amap for streaming results as they complete
        async with aiometer.amap(
            self.process_single_frame,
            frame_files,
            max_at_once=self.max_concurrent,
            max_per_second=max_per_second
        ) as result_stream:
            async for result in result_stream:
                if progress_callback:
                    progress_callback()

                if result.get('error'):
                    errors.append(result)
                results.append(result)

        await self.client.close()

        # Report errors summary
        if errors:
            click.echo(click.style(f"\n⚠️ {len(errors)} frames had errors:", fg='yellow'))
            for err in errors[:3]:  # Show first 3 errors
                click.echo(f"  Frame {err['frame']}: {err['error'][:50]}...")

        return results


# Utility functions for configuration (fallback to env vars for backwards compatibility)
def get_max_concurrent() -> int:
    """Get max concurrent setting from environment or default"""
    return int(os.getenv('OCR_MAX_CONCURRENT', '10'))


def get_max_per_second() -> float:
    """Get max requests per second from environment or default"""
    return float(os.getenv('OCR_MAX_PER_SECOND', '5'))