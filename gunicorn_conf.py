# gunicorn.conf.py
workers = 1
threads = 4
timeout = 120
bind = "0.0.0.0:10000"