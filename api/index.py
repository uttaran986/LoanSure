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
        matched_path = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI')
        path_info = environ.get('PATH_INFO', '')

        if path_info.startswith('/api/index'):
            if matched_path and not matched_path.startswith('/api/index'):
                environ['PATH_INFO'] = matched_path
            else:
                rest = path_info[len('/api/index'):]
                environ['PATH_INFO'] = rest if rest else '/'
        elif not path_info or path_info == '':
            environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
