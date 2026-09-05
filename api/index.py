import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class VercelPathFixMiddleware:
    """WSGI middleware to normalize PATH_INFO for Vercel Serverless routing."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # Extract original requested URI from all possible Vercel headers
        raw_uri = (
            environ.get('HTTP_X_MATCHED_PATH') or
            environ.get('HTTP_X_FORWARDED_URI') or
            environ.get('RAW_URI') or
            environ.get('REQUEST_URI') or
            environ.get('PATH_INFO', '')
        )
        
        # Remove query parameters if present
        if '?' in raw_uri:
            raw_uri = raw_uri.split('?')[0]

        if raw_uri.startswith('/api/index'):
            rest = raw_uri[len('/api/index'):]
            environ['PATH_INFO'] = rest if rest else '/'
        elif raw_uri and raw_uri != '':
            environ['PATH_INFO'] = raw_uri
        else:
            environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
