# Hugging Face Docker Space for the GSS MacroSystem Flask dashboard.
FROM python:3.11-slim

WORKDIR /app

# deps first (layer cache); gunicorn is already in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF runs the container as UID 1000. Big market/macro caches default to /tmp
# (writable), but the stock-sleeve CSV refreshes in-place under /app, so make the
# app tree writable too. Pin cache dirs to /tmp explicitly for clarity.
RUN chmod -R a+rwX /app
ENV GSS_CACHE_DIR=/tmp/gss_market_cache \
    GSS_MACRO_CACHE=/tmp/gss_macro_cache

EXPOSE 7860

# ONE worker on purpose: the app holds a single in-process cache + background
# stale-while-revalidate threads; multiple workers would duplicate memory and
# re-fetch yfinance N times. Threads add a little concurrency; the long timeout
# covers slow cold-start data pulls (yfinance/FRED/Zillow).
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:7860", "app:app"]
