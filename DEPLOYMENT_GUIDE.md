# NFL Betting Analyzer - Deployment & Production Guide

## 🚀 Production Deployment Guide

This comprehensive guide covers deploying the NFL Betting Analyzer system in production environments with high availability, scalability, and security.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Database Configuration](#database-configuration)
4. [Application Deployment](#application-deployment)
5. [API Server Setup](#api-server-setup)
6. [Monitoring & Logging](#monitoring--logging)
7. [Security Configuration](#security-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Backup & Recovery](#backup--recovery)
10. [Scaling Strategies](#scaling-strategies)
11. [Troubleshooting](#troubleshooting)

---

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 100GB SSD (500GB recommended)
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.8+ (3.9+ recommended)

### Recommended Production Setup
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 1TB+ NVMe SSD
- **Network**: 1Gbps+ bandwidth
- **Load Balancer**: Nginx/HAProxy
- **Database**: PostgreSQL 13+ cluster
- **Cache**: Redis cluster
- **Monitoring**: Prometheus + Grafana

---

## 🏗️ Infrastructure Setup

### 1. Server Provisioning

#### Single Server Setup (Development/Small Scale)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3.9 python3.9-venv python3-pip \
    postgresql postgresql-contrib redis-server nginx \
    supervisor git curl wget htop

# Create application user
sudo useradd -m -s /bin/bash nflanalyzer
sudo usermod -aG sudo nflanalyzer
```

#### Multi-Server Setup (Production)
```bash
# Application Servers (2+ instances)
# - NFL Analyzer API
# - Prediction Pipeline
# - Background Tasks

# Database Server
# - PostgreSQL Primary
# - PostgreSQL Replica (optional)

# Cache Server
# - Redis Master
# - Redis Replica (optional)

# Load Balancer
# - Nginx/HAProxy
# - SSL Termination
```

### 2. Docker Deployment (Recommended)

#### Create Docker Compose Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/nfl_predictions
      - REDIS_URL=redis://redis:6379/0
      - ENV=production
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=nfl_predictions
      - POSTGRES_USER=nfl_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "enhanced_prediction_api.py"]
```

---

## 🗄️ Database Configuration

### 1. PostgreSQL Setup

#### Installation and Configuration
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE nfl_predictions;
CREATE USER nfl_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE nfl_predictions TO nfl_user;
ALTER USER nfl_user CREATEDB;
EOF
```

#### Production PostgreSQL Configuration
```bash
# Edit /etc/postgresql/13/main/postgresql.conf
sudo nano /etc/postgresql/13/main/postgresql.conf
```

```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Connection settings
max_connections = 200
listen_addresses = '*'

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'
log_min_duration_statement = 1000
```

#### Database Initialization
```bash
# Run database setup
python scripts/setup_database.py --production

# Create indexes for performance
python -c "
from database_models import *
from sqlalchemy import create_engine, Index

engine = create_engine('postgresql://nfl_user:password@localhost/nfl_predictions')

# Create performance indexes
with engine.connect() as conn:
    conn.execute('CREATE INDEX IF NOT EXISTS idx_player_game_stats_player_id ON player_game_stats(player_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_player_game_stats_game_date ON player_game_stats(game_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON player_predictions(prediction_timestamp)')
    conn.commit()
"
```

### 2. Redis Configuration

#### Redis Setup for Caching
```bash
# Install Redis
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

```ini
# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Security
requirepass your_redis_password
bind 127.0.0.1

# Performance
tcp-keepalive 300
timeout 0
```

---

## 🚀 Application Deployment

### 1. Environment Setup

#### Create Production Environment File
```bash
# Create .env.production
cat > .env.production << EOF
# Environment
ENV=production
DEBUG=false

# Database
DATABASE_URL=postgresql://nfl_user:secure_password@localhost:5432/nfl_predictions

# Redis
REDIS_URL=redis://:redis_password@localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-super-secret-key-here
JWT_SECRET=your-jwt-secret-here

# External APIs
ODDS_API_KEY=your_odds_api_key
WEATHER_API_KEY=your_weather_api_key

# Logging
LOG_LEVEL=INFO
LOG_DIR=/var/log/nfl-analyzer

# Performance
MAX_WORKERS=8
BATCH_SIZE=500
CACHE_TTL=3600
EOF
```

### 2. Application Installation

#### Clone and Setup Application
```bash
# Switch to application user
sudo su - nflanalyzer

# Clone repository
git clone https://github.com/your-repo/nfl-betting-analyzer.git
cd nfl-betting-analyzer

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install production dependencies
pip install gunicorn supervisor psycopg2-binary

# Setup directories
mkdir -p logs data backups
chmod 755 logs data backups
```

#### Initialize Application
```bash
# Load environment
source .env.production

# Initialize database
python scripts/setup_database.py

# Download initial data
python scripts/download_initial_data.py

# Train initial models
python scripts/train_models.py

# Test system
python run_nfl_system.py status
```

---

## 🌐 API Server Setup

### 1. Gunicorn Configuration

#### Create Gunicorn Configuration
```python
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True

# Logging
accesslog = "/var/log/nfl-analyzer/gunicorn-access.log"
errorlog = "/var/log/nfl-analyzer/gunicorn-error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "nfl-analyzer-api"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
```

### 2. Supervisor Configuration

#### Create Supervisor Configuration
```ini
# /etc/supervisor/conf.d/nfl-analyzer.conf
[program:nfl-analyzer-api]
command=/home/nflanalyzer/nfl-betting-analyzer/venv/bin/gunicorn -c gunicorn.conf.py enhanced_prediction_api:app
directory=/home/nflanalyzer/nfl-betting-analyzer
user=nflanalyzer
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/nfl-analyzer/supervisor.log
environment=PATH="/home/nflanalyzer/nfl-betting-analyzer/venv/bin"

[program:nfl-analyzer-pipeline]
command=/home/nflanalyzer/nfl-betting-analyzer/venv/bin/python run_nfl_system.py run-pipeline
directory=/home/nflanalyzer/nfl-betting-analyzer
user=nflanalyzer
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/nfl-analyzer/pipeline.log
environment=PATH="/home/nflanalyzer/nfl-betting-analyzer/venv/bin"
```

#### Start Services
```bash
# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Start services
sudo supervisorctl start nfl-analyzer-api
sudo supervisorctl start nfl-analyzer-pipeline

# Check status
sudo supervisorctl status
```

### 3. Nginx Configuration

#### Create Nginx Configuration
```nginx
# /etc/nginx/sites-available/nfl-analyzer
upstream nfl_analyzer {
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Main application
    location / {
        proxy_pass http://nfl_analyzer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://nfl_analyzer;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Static files (if any)
    location /static/ {
        alias /home/nflanalyzer/nfl-betting-analyzer/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        access_log off;
        proxy_pass http://nfl_analyzer;
    }
}
```

#### Enable Site
```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/nfl-analyzer /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

---

## 📊 Monitoring & Logging

### 1. System Monitoring

#### Install Monitoring Tools
```bash
# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
sudo mv prometheus-*/prometheus /usr/local/bin/
sudo mv prometheus-*/promtool /usr/local/bin/

# Create Prometheus user and directories
sudo useradd --no-create-home --shell /bin/false prometheus
sudo mkdir /etc/prometheus /var/lib/prometheus
sudo chown prometheus:prometheus /etc/prometheus /var/lib/prometheus
```

#### Prometheus Configuration
```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "nfl_analyzer_rules.yml"

scrape_configs:
  - job_name: 'nfl-analyzer'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['localhost:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Application Metrics

#### Add Metrics Endpoint to API
```python
# Add to enhanced_prediction_api.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
REQUEST_COUNT = Counter('nfl_api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('nfl_api_request_duration_seconds', 'Request latency')
ACTIVE_PREDICTIONS = Gauge('nfl_active_predictions', 'Number of active predictions')
ERROR_COUNT = Counter('nfl_api_errors_total', 'Total API errors', ['error_type'])

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### 3. Log Management

#### Centralized Logging with ELK Stack
```bash
# Install Elasticsearch
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-7.x.list
sudo apt update && sudo apt install elasticsearch

# Install Logstash
sudo apt install logstash

# Install Kibana
sudo apt install kibana
```

#### Logstash Configuration
```ruby
# /etc/logstash/conf.d/nfl-analyzer.conf
input {
  file {
    path => "/var/log/nfl-analyzer/*.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  if [level] == "ERROR" {
    mutate {
      add_tag => ["error"]
    }
  }
  
  if [component] == "prediction" {
    mutate {
      add_tag => ["prediction"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "nfl-analyzer-%{+YYYY.MM.dd}"
  }
}
```

---

## 🔒 Security Configuration

### 1. SSL/TLS Setup

#### Generate SSL Certificates
```bash
# Using Let's Encrypt (recommended)
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Firewall Configuration

#### Configure UFW
```bash
# Enable firewall
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow specific services (if needed)
sudo ufw allow 5432  # PostgreSQL (only if remote access needed)
sudo ufw allow 6379  # Redis (only if remote access needed)

# Check status
sudo ufw status
```

### 3. Application Security

#### Environment Variables Security
```bash
# Secure environment file
sudo chmod 600 .env.production
sudo chown nflanalyzer:nflanalyzer .env.production

# Use secrets management (recommended for production)
# - AWS Secrets Manager
# - HashiCorp Vault
# - Kubernetes Secrets
```

---

## ⚡ Performance Optimization

### 1. Database Optimization

#### PostgreSQL Performance Tuning
```sql
-- Create materialized views for common queries
CREATE MATERIALIZED VIEW player_season_stats AS
SELECT 
    player_id,
    season,
    AVG(fantasy_points_ppr) as avg_fantasy_points,
    COUNT(*) as games_played
FROM player_game_stats pgs
JOIN games g ON pgs.game_id = g.game_id
GROUP BY player_id, season;

-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_player_predictions_composite 
ON player_predictions(player_id, game_id, prediction_timestamp);

-- Analyze tables
ANALYZE;
```

### 2. Application Optimization

#### Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    'prediction_cache_ttl': 3600,  # 1 hour
    'player_data_cache_ttl': 7200,  # 2 hours
    'market_data_cache_ttl': 300,   # 5 minutes
    'feature_cache_ttl': 1800,     # 30 minutes
}

# Connection pooling
DATABASE_POOL_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

### 3. System Optimization

#### System Tuning
```bash
# Increase file limits
echo "nflanalyzer soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "nflanalyzer hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65535" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## 💾 Backup & Recovery

### 1. Database Backup

#### Automated PostgreSQL Backup
```bash
#!/bin/bash
# /home/nflanalyzer/scripts/backup_db.sh

BACKUP_DIR="/home/nflanalyzer/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="nfl_predictions"

# Create backup
pg_dump -h localhost -U nfl_user -d $DB_NAME | gzip > "$BACKUP_DIR/nfl_predictions_$DATE.sql.gz"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "nfl_predictions_*.sql.gz" -mtime +7 -delete

# Upload to S3 (optional)
# aws s3 cp "$BACKUP_DIR/nfl_predictions_$DATE.sql.gz" s3://your-backup-bucket/
```

#### Schedule Backups
```bash
# Add to crontab
crontab -e
# Add: 0 2 * * * /home/nflanalyzer/scripts/backup_db.sh
```

### 2. Application Backup

#### Backup Script
```bash
#!/bin/bash
# /home/nflanalyzer/scripts/backup_app.sh

BACKUP_DIR="/home/nflanalyzer/backups"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/home/nflanalyzer/nfl-betting-analyzer"

# Create application backup
tar -czf "$BACKUP_DIR/app_backup_$DATE.tar.gz" \
    --exclude="venv" \
    --exclude="__pycache__" \
    --exclude="*.log" \
    --exclude="data/cache" \
    "$APP_DIR"

# Backup models
tar -czf "$BACKUP_DIR/models_backup_$DATE.tar.gz" "$APP_DIR/models"

# Keep only last 30 days
find $BACKUP_DIR -name "*_backup_*.tar.gz" -mtime +30 -delete
```

---

## 📈 Scaling Strategies

### 1. Horizontal Scaling

#### Load Balancer Configuration
```nginx
# Multiple application servers
upstream nfl_analyzer {
    least_conn;
    server 10.0.1.10:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 weight=1 max_fails=3 fail_timeout=30s;
}
```

### 2. Database Scaling

#### Read Replicas
```python
# Database routing configuration
DATABASE_CONFIG = {
    'write': 'postgresql://user:pass@primary-db:5432/nfl_predictions',
    'read': [
        'postgresql://user:pass@replica1-db:5432/nfl_predictions',
        'postgresql://user:pass@replica2-db:5432/nfl_predictions'
    ]
}
```

### 3. Microservices Architecture

#### Service Separation
```yaml
# docker-compose.microservices.yml
services:
  prediction-service:
    build: ./services/prediction
    ports:
      - "8001:8000"
    
  analytics-service:
    build: ./services/analytics
    ports:
      - "8002:8000"
    
  betting-service:
    build: ./services/betting
    ports:
      - "8003:8000"
    
  api-gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    depends_on:
      - prediction-service
      - analytics-service
      - betting-service
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Optimize Python memory
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2

# Restart services
sudo supervisorctl restart all
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Increase connection limit if needed
sudo nano /etc/postgresql/13/main/postgresql.conf
# max_connections = 200
```

#### 3. API Performance Issues
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"

# Monitor with htop
htop

# Check logs
tail -f /var/log/nfl-analyzer/gunicorn-error.log
```

#### 4. Cache Issues
```bash
# Check Redis status
redis-cli ping

# Clear cache if needed
redis-cli FLUSHALL

# Monitor Redis memory
redis-cli INFO memory
```

### Health Check Script
```bash
#!/bin/bash
# /home/nflanalyzer/scripts/health_check.sh

echo "=== NFL Analyzer Health Check ==="

# Check API
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
echo "API Status: $API_STATUS"

# Check Database
DB_STATUS=$(sudo -u postgres psql -d nfl_predictions -c "SELECT 1;" 2>/dev/null && echo "OK" || echo "FAIL")
echo "Database Status: $DB_STATUS"

# Check Redis
REDIS_STATUS=$(redis-cli ping 2>/dev/null || echo "FAIL")
echo "Redis Status: $REDIS_STATUS"

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
echo "Disk Usage: ${DISK_USAGE}%"

# Check memory
MEM_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
echo "Memory Usage: ${MEM_USAGE}%"

echo "=== Health Check Complete ==="
```

---

## 📞 Support and Maintenance

### Maintenance Schedule
- **Daily**: Automated backups, log rotation
- **Weekly**: Security updates, performance monitoring review
- **Monthly**: Full system backup, capacity planning review
- **Quarterly**: Security audit, disaster recovery testing

### Monitoring Alerts
Set up alerts for:
- API response time > 5 seconds
- Error rate > 5%
- Database connections > 80% of limit
- Disk usage > 85%
- Memory usage > 90%

### Contact Information
- **System Administrator**: admin@your-domain.com
- **Development Team**: dev@your-domain.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

---

**Last Updated**: 2024
**Version**: 2.0
**Environment**: Production Ready
