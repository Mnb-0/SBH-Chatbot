import asyncio
import os
import random
import aiofiles
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
    urls_to_crawl = [
        "https://salembalhamer.com/",
        "https://salembalhamer.com/company-overview/",
        "https://salembalhamer.com/industrial-sector/",
        "https://salembalhamer.com/trading-sector/",
        "https://salembalhamer.com/real-estate-sector/",
        "https://salembalhamer.com/services-sector/",
        "https://salembalhamercont.com/"
    ]
    
    if not os.path.exists("data"): 
        os.makedirs("data")

    # BROWSER CONFIG: Locked onto your new VPN-enabled work profile
    browser_config = BrowserConfig(
        headless=False, 
        user_data_dir="C:/Users/Mnb-0/AppData/Local/Google/Chrome/User Data",
        use_persistent_context=True,
        enable_stealth=True,
        extra_args=[
            "--profile-directory=Profile 1", 
            "--disable-blink-features=AutomationControlled"
        ]
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        delay_before_return_html=2.0,
        word_count_threshold=10,
        exclude_external_links=True,
        excluded_tags=["header", "footer", "nav", "aside", "form"] 
    )

    scraped_urls = set(urls_to_crawl) 
    queue_list = list(urls_to_crawl)

    print("Starting sequential crawler on Profile 1...")

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while queue_list:
                current_url = queue_list.pop(0)
                
                safe_name = current_url.split("://")[-1].replace("/", "_").replace(".", "_").strip("_")
                if not safe_name: safe_name = "homepage"
                
                # Resumes automatically if it crashes
                if os.path.exists(f"data/{safe_name}.md"):
                    print(f"⏭️ Skipping (already downloaded): {safe_name}")
                    continue

                print(f"🚀 Ingesting: {current_url}")
                
                try:
                    result = await crawler.arun(url=current_url, config=run_config)

                    if result.success:
                        async with aiofiles.open(f"data/{safe_name}.md", "w", encoding="utf-8") as f:
                            await f.write(result.markdown)
                        
                        if result.links and "internal" in result.links:
                            for link_obj in result.links["internal"]:
                                href = link_obj.get("href")
                                if href and ("salembalhamer" in href) and (href not in scraped_urls):
                                    href_lower = href.lower()
                                    if "/ar/" in href_lower or href_lower.endswith("/ar") or "-ar" in href_lower:
                                        continue 
                                    scraped_urls.add(href) 
                                    queue_list.append(href)
                        
                        print(f"✅ Saved {safe_name}. Queue size: {len(queue_list)}")
                        
                        delay = random.uniform(3.0, 7.0)
                        print(f"⏳ Sleeping for {delay:.2f} seconds...\n")
                        await asyncio.sleep(delay)

                    else:
                        print(f"❌ Error on {current_url}: {result.error_message}\n")
                        await asyncio.sleep(3)
                        
                except Exception as e:
                    print(f"🔥 Hard Crash on {current_url}: {str(e)}\n")
                    await asyncio.sleep(5) 
                    
    except Exception as fatal_error:
        print(f"\n💀 FATAL CRASH: {fatal_error}\nYou likely left Profile 1 open. Close it and re-run.")

if __name__ == "__main__":
    asyncio.run(main())