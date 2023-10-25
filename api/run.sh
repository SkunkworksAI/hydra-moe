# Add gunicorn logging configurations

export HOST="localhost"
export PORT="8000"
export ACCESSLOG="-"  # '-' means log to stdout
export ERRORLOG="-"  # '-' means log to stderr
export LOGLEVEL="info"  # Set log level to info
export APP_MODULE="hydramoe_api.server:app"
# Run gunicorn
exec gunicorn --worker-tmp-dir /dev/shm --workers=6 \
    --bind $HOST:$PORT "$APP_MODULE" -k uvicorn.workers.UvicornWorker \
    --access-logfile $ACCESSLOG --error-logfile $ERRORLOG --log-level $LOGLEVEL --timeout 1000 \