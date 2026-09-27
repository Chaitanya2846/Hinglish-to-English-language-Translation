import os
import sys
import json
import hashlib
import re
from datetime import datetime
from dateutil import tz

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'datasets_log.jsonl')

PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\+?\d[\d\-\s]{7,}\d"),
]

def _hash_pii(match: re.Match) -> str:
    token = match.group(0)
    return "__PII_" + hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]

def scrub_pii(text: str) -> str:
    for patt in PII_PATTERNS:
        text = patt.sub(_hash_pii, text)
    return text

def log_batch(meta: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or raw ID."""
    s = url_or_id.strip()
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|\/|$)", s)
    if match:
        return match.group(1)
    if len(s) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", s):
        return s
    return s

def save_raw_batch(records, source_name, method='youtube_api'):
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(base, exist_ok=True)
    timestamp = datetime.now(tz.tzutc()).strftime('%Y%m%dT%H%M%SZ')
    filename = f"raw_{source_name}_{timestamp}.jsonl"
    path = os.path.join(base, filename)
    count = 0
    with open(path, 'w', encoding='utf-8') as fw:
        for r in records:
            rec = dict(r)
            if 'text' in rec and isinstance(rec['text'], str):
                rec['text'] = scrub_pii(rec['text'])
            if 'author' in rec and isinstance(rec['author'], str):
                rec['author_hashed'] = hashlib.sha256(rec['author'].encode('utf-8')).hexdigest()
                rec.pop('author', None)
            fw.write(json.dumps(rec, ensure_ascii=False) + '\n')
            count += 1

    meta = {
        'source': source_name,
        'method': method,
        'date_collected': timestamp,
        'file': os.path.relpath(path, os.path.join(os.path.dirname(__file__), '..')),
        'size': count,
        'license': 'platform_terms',
    }
    log_batch(meta)
    return path

def scrape_video_comments(video_ids, api_key=None, max_results=100):
    """Scrape comments for a list of YouTube video IDs using the official Data API.

    Returns list of records: {source, text, timestamp, source_id, author}
    """
    if not api_key:
        api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise RuntimeError('YouTube API key missing; provide api_key or set YOUTUBE_API_KEY in .env')

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except Exception:
        raise RuntimeError('google-api-python-client not installed')

    youtube = build('youtube', 'v3', developerKey=api_key)
    records = []
    clean_vids = [extract_video_id(v) for v in video_ids]

    for vid in clean_vids:
        print(f"Scraping comments for video: {vid} (up to {max_results})...")
        collected = 0
        next_page_token = None
        while True:
            to_fetch = min(100, max_results - collected)
            if to_fetch <= 0:
                break
            try:
                req = youtube.commentThreads().list(
                    part='snippet',
                    videoId=vid,
                    textFormat='plainText',
                    maxResults=to_fetch,
                    pageToken=next_page_token
                )
                resp = req.execute()
            except HttpError as e:
                print(f"  [!] Skipped video {vid}: {e.reason}")
                break

            items = resp.get('items', [])
            if not items:
                break

            for item in items:
                snip = item['snippet']['topLevelComment']['snippet']
                rec = {
                    'source': f'youtube/{vid}',
                    'text': snip.get('textDisplay', ''),
                    'timestamp': snip.get('publishedAt', ''),
                    'source_id': item.get('id'),
                    'author': snip.get('authorDisplayName', '')
                }
                records.append(rec)
                collected += 1

            next_page_token = resp.get('nextPageToken')
            if not next_page_token:
                break

        print(f"  -> Collected {collected} comments for {vid}")

    return records

def search_videos(api_key, query, max_results=5, regionCode='IN'):
    """Search YouTube for videos matching a query. Returns list of (videoId, title)."""
    if not api_key:
        api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise RuntimeError('YouTube API key missing; provide api_key or set YOUTUBE_API_KEY in .env')

    try:
        from googleapiclient.discovery import build
    except Exception:
        raise RuntimeError('google-api-python-client not installed')

    youtube = build('youtube', 'v3', developerKey=api_key)
    req = youtube.search().list(part='snippet', q=query, maxResults=max_results, type='video', regionCode=regionCode)
    resp = req.execute()
    out = []
    for item in resp.get('items', []):
        vid = item['id']['videoId']
        title = item['snippet'].get('title', '')
        out.append((vid, title))
    return out

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Scrape Hinglish comments from YouTube videos.')
    p.add_argument('--videos', nargs='+', help='List of video IDs or YouTube URLs')
    p.add_argument('--search', type=str, help='Search query to discover relevant Indian/Hinglish videos (e.g. "standup comedy hindi")')
    p.add_argument('--max_videos', type=int, default=3, help='Max videos to fetch when using --search')
    p.add_argument('--max_results', type=int, default=100, help='Max comments per video (default 100)')
    p.add_argument('--api_key', help='YouTube Data API v3 key (defaults to YOUTUBE_API_KEY from .env)')
    p.add_argument('--tag', type=str, default='youtube', help='Identifier tag in saved filename')
    args = p.parse_args()

    api_key = args.api_key or os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("ERROR: No YouTube API key provided. Set YOUTUBE_API_KEY in .env or pass --api_key")
        exit(1)

    target_videos = []
    if args.videos:
        target_videos.extend(args.videos)

    if args.search:
        print(f"Searching YouTube for '{args.search}' in region IN...")
        results = search_videos(api_key, args.search, max_results=args.max_videos)
        for vid, title in results:
            print(f"  [Found] {vid} - {title}")
            target_videos.append(vid)

    if not target_videos:
        print("ERROR: Please specify --videos or --search")
        exit(1)

    recs = scrape_video_comments(target_videos, api_key=api_key, max_results=args.max_results)
    if recs:
        path = save_raw_batch(recs, args.tag)
        print(f"\n[+] Successfully saved {len(recs)} raw comments to {path}")
    else:
        print("\n[-] No comments were collected.")
