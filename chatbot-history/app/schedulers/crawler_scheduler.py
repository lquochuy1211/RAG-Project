# app/scheduler/crawler_scheduler.py

import logging
import threading
from datetime import datetime, time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from app.data_ingestion.enhanced_pipeline import EnhancedIngestionPipeline
from app.data_ingestion.auto_crawler import AutoCrawler
from app.data_ingestion.wikipedia_scraper import WikipediaScraper
from app.data_ingestion.data_processor import DataProcessor
from app.db.qdrant_client import get_client, init_collection
from app.config.settings import settings
from qdrant_client.models import PointStruct
import uuid

logger = logging.getLogger(__name__)

# Global scheduler
scheduler = None
scheduler_lock = threading.Lock()


def get_scheduler() -> BackgroundScheduler:
    """Get or create scheduler instance."""
    global scheduler

    if scheduler is None:
        with scheduler_lock:
            if scheduler is None:
                scheduler = BackgroundScheduler(daemon=True)

    return scheduler


def run_daily_crawl():
    """Daily crawl job - discover + ingest new articles"""

    logger.info("=" * 80)
    logger.info(f"[SCHEDULER] DAILY CRAWL - {datetime.now()}")
    logger.info("=" * 80)

    try:
        # Step 1: Init
        logger.info("[STEP 1/6] Initializing...")
        init_collection()

        # Step 2: Discover new articles
        logger.info("[STEP 2/6] Discovering new articles...")
        crawler = AutoCrawler()
        new_articles = crawler.discover_new_articles(max_per_category=50)

        if not new_articles:
            logger.warning("❌ No new articles found")
            logger.info("=" * 80)
            return

        logger.info(f"✅ Found {len(new_articles)} new articles")

        # Step 3: Scrape articles
        logger.info(f"[STEP 3/6] Scraping {len(new_articles)} articles...")
        scraper = WikipediaScraper()
        articles = scraper.fetch_articles(new_articles)

        logger.info(f"✅ Scraped {len(articles)}/{len(new_articles)} articles")

        # Step 4: Process articles
        logger.info("[STEP 4/6] Processing articles...")
        processor = DataProcessor()
        all_chunks = []

        for idx, article in enumerate(articles, 1):
            chunks = processor.process_article(article)
            all_chunks.extend(chunks)
            logger.debug(f"  [{idx}/{len(articles)}] {article.get('title', 'Unknown')} -> {len(chunks)} chunks")

        logger.info(f"✅ Created {len(all_chunks)} chunks")

        # Step 5: Save to Qdrant
        logger.info("[STEP 5/6] Saving to Qdrant...")
        qdrant = get_client()

        points = []
        for chunk in all_chunks:
            payload = {
                "text": chunk["text"],
                "title": chunk["title"],
                "url": chunk["url"],
                "source": chunk["source"],
                "chunk_id": chunk.get("chunk_id", 0),
                "content_hash": chunk.get("content_hash", ""),
                "ingestion_date": datetime.utcnow().isoformat(),

                # ✅ Backend fields
                "group_id": "PUBLIC",
                "tenancy": "PUBLIC",
                "is_shared": True,
                "message_type": "knowledge",
            }

            points.append(PointStruct(
                id=str(chunk.get("id", uuid.uuid4())),
                vector=chunk["vector"],
                payload=payload
            ))

        qdrant.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        logger.info(f"✅ Saved {len(points)} chunks")

        # Step 6: Mark as crawled
        logger.info("[STEP 6/6] Marking as crawled...")
        crawler.mark_as_crawled(new_articles)
        logger.info(f"✅ Marked {len(new_articles)} articles as crawled")

        # Get stats
        count_result = qdrant.count(collection_name=settings.COLLECTION_NAME)

        logger.info("=" * 80)
        logger.info("[SUCCESS] DAILY CRAWL COMPLETED")
        logger.info(f"  - Articles: {len(articles)}")
        logger.info(f"  - Chunks: {len(all_chunks)}")
        logger.info(f"  - Total in DB: {count_result.count}")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[ERROR] Daily crawl failed: {e}")
        logger.info("=" * 80)


def start_scheduler(enable_daily_crawl: bool = True, crawl_hour: int = 2):
    """
    Start background scheduler.

    Args:
        enable_daily_crawl: Enable daily crawl job (default: True)
        crawl_hour: Hour to run crawl (0-23, default: 2 = 2 AM)
    """

    scheduler_instance = get_scheduler()

    if scheduler_instance.running:
        logger.warning("[SCHEDULER] Scheduler already running")
        return

    logger.info("=" * 80)
    logger.info("[SCHEDULER] Starting background scheduler")
    logger.info("=" * 80)

    # Add daily crawl job
    if enable_daily_crawl:
        scheduler_instance.add_job(
            run_daily_crawl,
            trigger=CronTrigger(hour=crawl_hour, minute=0),  # 2 AM daily
            id='daily_crawl',
            name='Daily Crawler Job',
            replace_existing=True,
            max_instances=1,  # Chỉ chạy 1 instance
        )
        logger.info(f"✅ Daily crawl scheduled at {crawl_hour}:00 AM")

    # Start scheduler
    scheduler_instance.start()
    logger.info("[SCHEDULER] ✅ Scheduler started")
    logger.info("=" * 80)


def stop_scheduler():
    """Stop scheduler gracefully."""

    scheduler_instance = get_scheduler()

    if scheduler_instance.running:
        logger.info("[SCHEDULER] Stopping scheduler...")
        scheduler_instance.shutdown(wait=True)
        logger.info("[SCHEDULER] ✅ Scheduler stopped")
    else:
        logger.info("[SCHEDULER] Scheduler not running")


def get_scheduled_jobs():
    """Get list of scheduled jobs."""

    scheduler_instance = get_scheduler()
    jobs = scheduler_instance.get_jobs()

    jobs_info = []
    for job in jobs:
        jobs_info.append({
            "id": job.id,
            "name": job.name,
            "trigger": str(job.trigger),
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return jobs_info
