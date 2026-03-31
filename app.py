import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, redirect
from datetime import datetime, date, timedelta
import awswrangler as wr
import boto3
import traceback
import math
import time
import requests

# --- AI AGENT IMPORT ---
# from agent.router import ask_vibe_agent

# --- PLOTLY & BOKEH IMPORTS ---
from sklearn.linear_model import LinearRegression
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use('Agg')
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import json_item

import psycopg2
from psycopg2.extras import execute_values
from contextlib import contextmanager
from flask import session, url_for

# --- AUTH MODULE ---
from auth import (
    authenticate_user, register_user, login_required, role_required,
    get_user_permissions, get_all_users, get_login_history,
    update_user, delete_user, change_password
)

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', 'vibe-production-secret-key-2026')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# --- POSTGRES DB CONFIG ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'vibe_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'port': os.getenv('DB_PORT', '5432')
}

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# --- AWS ATHENA CONFIGURATION ---
ATHENA_DATABASE = "advanced-analytics"
S3_STAGING_DIR = "s3://neo-advanced-analytics/athena-query-results/"
PRICING_FILE = 'capex_pricing.json'

ATHENA_CACHE_SETTINGS = {
    "max_cache_seconds": 604800, # Cache for 7 Days
    "max_cache_query_inspections": 500
}

RAM_CACHE = {}

def api_login_required(f):
    """Decorator for API routes that returns JSON instead of redirecting"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function
    
def get_cached_dataframe(sql):
    """Fetches from server RAM if available, otherwise asks Athena/S3."""
    now = time.time()
    
    # 1. Check if the exact SQL is already in RAM and less than 7 days old
    if sql in RAM_CACHE and (now - RAM_CACHE[sql]['timestamp']) < 604800:
        return RAM_CACHE[sql]['df']
        
    # 2. If not in RAM, use Wrangler (which will use the S3 cache if available!)
    df = wr.athena.read_sql_query(
        sql=sql, 
        database=ATHENA_DATABASE, 
        s3_output=S3_STAGING_DIR, 
        boto3_session=aws_session, 
        ctas_approach=False, 
        athena_cache_settings=ATHENA_CACHE_SETTINGS
    )
    
    # 3. Store the full dataset in RAM for the next pagination click
    RAM_CACHE[sql] = {'timestamp': now, 'df': df}
    return df

# [CRITICAL FIX]: Force the exact AWS Region so Wrangler doesn't get lost, 
# and disable CTAS to prevent bucket verification errors.
aws_session = boto3.Session(region_name="ap-southeast-1")

# --- MALAYSIA HOLIDAYS ---
MALAYSIA_HOLIDAYS = {
    datetime(2026, 1, 1): "New Year", datetime(2026, 2, 1): "Federal Territory",
    datetime(2026, 2, 17): "CNY", datetime(2026, 3, 20): "Hari Raya Aidilfitri",
    datetime(2026, 5, 1): "Labour Day", datetime(2026, 5, 27): "Hari Raya Haji",
    datetime(2026, 8, 31): "Merdeka", datetime(2026, 9, 16): "Malaysia Day",
    datetime(2026, 12, 25): "Christmas"
}

def apply_pandas_filters(df, request_args):
    """Filters a loaded Pandas DataFrame based on UI request arguments in memory."""
    if df.empty:
        return df
        
    filtered_df = df.copy()

    # Filter by Region
    region = request_args.get('region')
    if region and region != 'All' and 'region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['region'].str.upper() == region.upper()]

    # Filter by Operator
    operator = request_args.get('operator')
    if operator and operator != 'All' and 'operator' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['operator'] == operator]

    # Filter by Cluster
    cluster = request_args.get('cluster')
    if cluster and cluster != 'All' and 'cluster' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['cluster'] == cluster]

    # Filter by Week
    week = request_args.get('week')
    if week and str(week).lower() not in ['all', ''] and 'week' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['week'].astype(str) == str(week)]

    return filtered_df
@app.route('/api/map/upgrade-cases')
@api_login_required
def api_map_upgrade_cases():
    """Get sites with upgrade cases directly from Athena"""
    week = request.args.get('week', type=int)
    year = request.args.get('year', str(datetime.now().year))
    
    if not week:
        return jsonify([]), 400
    
    try:
        sql = f"""
            SELECT DISTINCT
                split_part(cu.zoom_sector_id, '_', 1) as site_id,
                cu.zoom_sector_id,
                cu.suggested_upgrade_case as upgrade_case,
                cu.estimated_total_capex_rm as total_capex,
                cu.projected_prb_pct as prb,
                ca.eric_dl_user_ip_thpt as dl_thpt,
                GREATEST(COALESCE(ca.eric_max_rrc_user,0), COALESCE(ca.max_active_user,0)) as user_count,
                CAST(cu.data_week AS INTEGER) as week
            FROM capex_upgrades cu
            LEFT JOIN congestion_analysis ca 
                ON cu.zoom_sector_id = ca.zoom_sector_id 
                AND cu.data_week = ca.week
                AND CAST(ca.year AS VARCHAR) = '{year}'
            WHERE cu.suggested_upgrade_case IS NOT NULL
              AND cu.suggested_upgrade_case NOT IN ('', 'None', 'No Upgrade Needed')
              AND CAST(cu.data_week AS INTEGER) = {week}
            ORDER BY cu.estimated_total_capex_rm DESC
        """
        
        df = get_cached_dataframe(sql)
        
        if df.empty:
            return jsonify([])
        
        # Group by site_id
        result = []
        for site_id, group in df.groupby('site_id'):
            upgrade_details = []
            for _, row in group.iterrows():
                upgrade_details.append({
                    'sector_id': row['zoom_sector_id'],
                    'upgrade_case': row['upgrade_case'],
                    'capex': float(row['total_capex']) if pd.notna(row['total_capex']) else 0,
                    'prb': float(row['prb']) if pd.notna(row['prb']) else 0,
                    'thpt': float(row['dl_thpt']) if pd.notna(row['dl_thpt']) else 0,
                    'users': int(row['user_count']) if pd.notna(row['user_count']) else 0
                })
            
            result.append({
                'site_id': site_id,
                'upgrade_details': upgrade_details,
                'total_capex': sum(d['capex'] for d in upgrade_details)
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error fetching upgrade cases: {e}")
        traceback.print_exc()
        return jsonify([]), 500
        
# --- CORE ROUTES ---
@app.route('/')
@api_login_required
def index(): 
    role = session.get('role', 'Staff')
    return render_template(
        'index.html', 
        user_id=session.get('user_id'), 
        username=session.get('username', 'User'), 
        full_name=session.get('full_name', ''), 
        role=role
    )

@app.route('/map')
@api_login_required
def map_view():
    role = session.get('role', 'Staff')
    
    # 1. Fetch the permissions for this specific role
    user_permissions = get_user_permissions(role) 
    
    return render_template(
        'map.html',
        user_id=session.get('user_id'),
        username=session.get('username', 'User'),
        full_name=session.get('full_name', ''),
        role=role,
        permissions=user_permissions  # 2. Pass it to the template!
    )

@app.route('/iam')
@api_login_required
@role_required('Admin')
def iam_panel():
    role = session.get('role', 'Admin')
    return render_template(
        'iam.html', 
        user_id=session.get('user_id'), 
        username=session.get('username', 'Admin'), 
        full_name=session.get('full_name', ''), 
        role=role
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password: return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'Unknown')
    success, user_data, message = authenticate_user(username, password, ip_address, user_agent)
    
    if success:
        session['user_id'] = user_data['id']
        session['username'] = user_data['username']
        session['role'] = user_data['role']
        session['full_name'] = user_data['full_name']
        session.permanent = True
        return jsonify({'success': True, 'message': message, 'redirect': '/'})
    return jsonify({'success': False, 'message': message}), 401

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()
    full_name = data.get('full_name', '').strip()
    role = data.get('role', 'Staff')
    
    if not all([username, password, email, full_name]): return jsonify({'success': False, 'message': 'All fields are required'}), 400
    success, message = register_user(username, password, email, full_name, role)
    if success:
        return jsonify({'success': True, 'message': message, 'redirect': '/login'})
    return jsonify({'success': False, 'message': message}), 400

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/iam/users', methods=['GET'])
@api_login_required
@role_required('Admin')
def get_users():
    return jsonify(get_all_users())

@app.route('/api/iam/users/<int:user_id>', methods=['PUT', 'DELETE'])
@api_login_required
@role_required('Admin')
def manage_user(user_id):
    if request.method == 'PUT':
        success, message = update_user(user_id, **request.json)
    else:
        success, message = delete_user(user_id)
    return jsonify({'success': success, 'message': message})

@app.route('/api/iam/login-history', methods=['GET'])
@api_login_required
@role_required('Admin')
def get_login_history_route():
    return jsonify(get_login_history())

@app.route('/api/iam/activity', methods=['GET'])
@api_login_required
@role_required('Admin')
def get_user_activity():
    filter_type = request.args.get('filter', 'all')
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 20, type=int)

    try:
        with get_db_connection() as conn:
            parts, params = [], []
            if filter_type in ('all', 'annotation'):
                parts.append("""SELECT 'annotation' AS type, ma.created_by_username AS username, ma.created_at AS timestamp, ma.title AS title, ma.shape_type AS shape_type, ma.priority AS priority, ma.status AS ann_status, NULL::TEXT AS partner_name, NULL::TEXT AS preview FROM map_annotations ma""")
            if filter_type in ('all', 'message'):
                parts.append("""SELECT 'message' AS type, sender.username AS username, m.sent_at AS timestamp, NULL::TEXT AS title, NULL::TEXT AS shape_type, NULL::TEXT AS priority, NULL::TEXT AS ann_status, partner.username AS partner_name, LEFT(m.content, 80) AS preview FROM messages m JOIN users sender ON m.sender_id = sender.id JOIN conversations c ON m.conversation_id = c.id JOIN conversation_participants cp ON cp.conversation_id = c.id AND cp.user_id != m.sender_id JOIN users partner ON cp.user_id = partner.id""")
            if not parts: return jsonify([])
            
            final_sql = f"SELECT * FROM ({' UNION ALL '.join(parts)}) AS activity ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            params += [limit, offset]
            df = pd.read_sql(final_sql, conn, params=params)
            df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat() if pd.notna(x) and x is not None else None)
            return jsonify(df.replace({float('nan'): None}).to_dict('records'))
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/user/permissions', methods=['GET'])
@api_login_required
def get_permissions():
    return jsonify(get_user_permissions(session.get('role', 'Staff')))

@app.route('/api/user/change-password', methods=['POST'])
@api_login_required
def change_user_password():
    new_password = request.json.get('new_password', '')
    if not new_password or len(new_password) < 6: return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
    success, message = change_password(session.get('user_id'), new_password)
    return jsonify({'success': success, 'message': message})

@app.route('/api/user/profile', methods=['GET', 'PUT'])
@api_login_required
def user_profile():
    user_id = session.get('user_id')
    if request.method == 'GET':
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, full_name, role FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
            if not row: return jsonify({'error': 'User not found'}), 404
            return jsonify({'id': row[0], 'username': row[1], 'email': row[2], 'full_name': row[3], 'role': row[4]})
    
    data = request.json
    full_name, email = data.get('full_name', '').strip(), data.get('email', '').strip()
    if not full_name or not email: return jsonify({'success': False, 'message': 'Full name and email are required'}), 400
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE email = %s AND id != %s", (email, user_id))
            if cursor.fetchone(): return jsonify({'success': False, 'message': 'Email already in use'}), 400
            cursor.execute("UPDATE users SET full_name = %s, email = %s WHERE id = %s", (full_name, email, user_id))
        session['full_name'] = full_name
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e: return jsonify({'success': False, 'message': str(e)}), 500

def get_or_create_conversation(user_id, other_user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cp1.conversation_id FROM conversation_participants cp1
            JOIN conversation_participants cp2 ON cp1.conversation_id = cp2.conversation_id
            JOIN conversations c ON c.id = cp1.conversation_id
            WHERE cp1.user_id = %s AND cp2.user_id = %s AND c.is_group = FALSE
        """, (user_id, other_user_id))
        row = cursor.fetchone()
        if row: return row[0]
        cursor.execute("INSERT INTO conversations (created_by, is_group) VALUES (%s, FALSE) RETURNING id", (user_id,))
        conv_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO conversation_participants (conversation_id, user_id) VALUES (%s, %s), (%s, %s)", (conv_id, user_id, conv_id, other_user_id))
        return conv_id

@app.route('/api/messages/conversations', methods=['GET'])
@api_login_required
def get_conversations():
    user_id = session.get('user_id')
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.title, c.is_group,
                    ARRAY_AGG(u.full_name) FILTER (WHERE u.id != %s) AS member_names,
                    ARRAY_AGG(u.username)  FILTER (WHERE u.id != %s) AS member_usernames,
                    (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY sent_at DESC LIMIT 1) AS last_message,
                    (SELECT sent_at FROM messages WHERE conversation_id = c.id ORDER BY sent_at DESC LIMIT 1) AS last_time,
                    (SELECT COUNT(*) FROM messages m2 WHERE m2.conversation_id = c.id AND m2.sender_id != %s
                     AND NOT EXISTS (SELECT 1 FROM message_reads mr WHERE mr.message_id = m2.id AND mr.user_id = %s)) AS unread_count
                FROM conversations c
                JOIN conversation_participants cp  ON cp.conversation_id  = c.id AND cp.user_id = %s
                JOIN conversation_participants cp2 ON cp2.conversation_id = c.id
                JOIN users u ON u.id = cp2.user_id
                GROUP BY c.id, c.title, c.is_group
                ORDER BY last_time DESC NULLS LAST
            """, (user_id, user_id, user_id, user_id, user_id))
            result = []
            for r in cursor.fetchall():
                display_name = r[1] or 'Group Chat' if r[2] else (r[3][0] if r[3] else 'Unknown')
                result.append({'id': r[0], 'title': display_name, 'is_group': r[2], 'member_names': r[3] or [], 'member_usernames': r[4] or [], 'partner_name': display_name, 'last_message': r[5], 'last_time': r[6].isoformat() if r[6] else None, 'unread_count': int(r[7])})
            return jsonify(result)
    except Exception: return jsonify([])

@app.route('/api/messages/conversation/<int:conv_id>', methods=['GET'])
@api_login_required
def get_conversation_messages(conv_id):
    user_id = session.get('user_id')
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM conversation_participants WHERE conversation_id = %s AND user_id = %s", (conv_id, user_id))
        if not cursor.fetchone(): return jsonify({'error': 'Unauthorized'}), 403
        cursor.execute("""
            INSERT INTO message_reads (message_id, user_id)
            SELECT m.id, %s FROM messages m WHERE m.conversation_id = %s AND m.sender_id != %s AND NOT EXISTS (SELECT 1 FROM message_reads mr WHERE mr.message_id = m.id AND mr.user_id = %s) ON CONFLICT DO NOTHING
        """, (user_id, conv_id, user_id, user_id))
        cursor.execute("SELECT m.id, m.sender_id, u.full_name, m.content, m.sent_at, (m.sender_id = %s) FROM messages m JOIN users u ON u.id = m.sender_id WHERE m.conversation_id = %s ORDER BY m.sent_at ASC", (user_id, conv_id))
        return jsonify([{'id': r[0], 'sender_id': r[1], 'sender_name': r[2], 'content': r[3], 'sent_at': r[4].isoformat(), 'is_mine': r[5]} for r in cursor.fetchall()])

@app.route('/api/messages/send', methods=['POST'])
@api_login_required
def send_message():
    user_id, data = session.get('user_id'), request.json
    conv_id, content = data.get('conversation_id'), data.get('content', '').strip()
    if not conv_id or not content: return jsonify({'success': False}), 400
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM conversation_participants WHERE conversation_id = %s AND user_id = %s", (conv_id, user_id))
        if not cursor.fetchone(): return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        cursor.execute("INSERT INTO messages (conversation_id, sender_id, content) VALUES (%s, %s, %s)", (conv_id, user_id, content))
        cursor.execute("INSERT INTO message_reads (message_id, user_id) SELECT currval('messages_id_seq'), %s ON CONFLICT DO NOTHING", (user_id,))
        return jsonify({'success': True})

@app.route('/api/messages/new', methods=['POST'])
@api_login_required
def start_new_conversation():
    user_id, data = session.get('user_id'), request.json
    recipient_id, content = data.get('recipient_id'), data.get('content', '').strip()
    if not recipient_id or not content or recipient_id == user_id: return jsonify({'success': False}), 400
    conv_id = get_or_create_conversation(user_id, recipient_id)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (conversation_id, sender_id, content) VALUES (%s, %s, %s)", (conv_id, user_id, content))
        cursor.execute("SELECT full_name FROM users WHERE id = %s", (recipient_id,))
        return jsonify({'success': True, 'conversation_id': conv_id, 'partner_name': cursor.fetchone()[0]})

@app.route('/api/messages/group/new', methods=['POST'])
@api_login_required
def start_group_conversation():
    user_id, data = session.get('user_id'), request.json
    member_ids = data.get('member_ids', [])
    if len(member_ids) < 2: return jsonify({'success': False}), 400
    title = data.get('title', '').strip() or 'Group Chat'
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (title, created_by, is_group) VALUES (%s, %s, TRUE) RETURNING id", (title, user_id))
        conv_id = cursor.fetchone()[0]
        for uid in list(set([user_id] + member_ids)):
            cursor.execute("INSERT INTO conversation_participants (conversation_id, user_id, is_admin) VALUES (%s, %s, %s)", (conv_id, uid, uid == user_id))
        return jsonify({'success': True, 'conversation_id': conv_id, 'title': title})

@app.route('/api/messages/group/<int:conv_id>/<action>', methods=['POST'])
@api_login_required
def manage_group(conv_id, action):
    user_id, data = session.get('user_id'), request.json or {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_group FROM conversations WHERE id = %s", (conv_id,))
        if not cursor.fetchone()[0]: return jsonify({'success': False}), 400
        cursor.execute("SELECT is_admin FROM conversation_participants WHERE conversation_id = %s AND user_id = %s", (conv_id, user_id))
        admin_check = cursor.fetchone()
        if action in ['add', 'remove', 'rename', 'delete'] and not (admin_check and admin_check[0]): return jsonify({'success': False}), 403
        
        if action == 'leave': cursor.execute("DELETE FROM conversation_participants WHERE conversation_id = %s AND user_id = %s", (conv_id, user_id))
        elif action == 'add': cursor.execute("INSERT INTO conversation_participants (conversation_id, user_id, is_admin) VALUES (%s, %s, FALSE) ON CONFLICT DO NOTHING", (conv_id, data.get('user_id')))
        elif action == 'remove': cursor.execute("DELETE FROM conversation_participants WHERE conversation_id = %s AND user_id = %s", (conv_id, data.get('user_id')))
        elif action == 'rename': cursor.execute("UPDATE conversations SET title = %s WHERE id = %s", (data.get('title'), conv_id))
        elif action == 'delete': cursor.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        return jsonify({'success': True})

@app.route('/api/messages/group/<int:conv_id>/members', methods=['GET'])
@api_login_required
def get_group_members(conv_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT u.id, u.full_name, u.username, u.role, cp.is_admin, cp.joined_at FROM conversation_participants cp JOIN users u ON u.id = cp.user_id WHERE cp.conversation_id = %s ORDER BY cp.is_admin DESC, cp.joined_at ASC", (conv_id,))
        return jsonify([{'id': r[0], 'full_name': r[1], 'username': r[2], 'role': r[3], 'is_admin': r[4]} for r in cursor.fetchall()])

@app.route('/api/messages/users', methods=['GET'])
@api_login_required
def get_users_for_messaging():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, username FROM users WHERE is_active = TRUE ORDER BY full_name")
        return jsonify([{'id': r[0], 'full_name': r[1], 'username': r[2]} for r in cursor.fetchall()])

@app.route('/api/messages/unread-count', methods=['GET'])
@@api_login_required
def get_unread_count():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT COUNT(*) FROM messages m JOIN conversation_participants cp ON cp.conversation_id = m.conversation_id WHERE cp.user_id = %s AND m.sender_id != %s AND NOT EXISTS (SELECT 1 FROM message_reads mr WHERE mr.message_id = m.id AND mr.user_id = %s)""", (session.get('user_id'), session.get('user_id'), session.get('user_id')))
        return jsonify({'count': cursor.fetchone()[0]})

@app.route('/api/reviews', methods=['GET', 'POST'])
@api_login_required
def handle_reviews():
    if request.method == 'GET':
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT id, user_id, username, category, rating, title, body, is_anonymous, created_at, updated_at FROM reviews"
            params = []
            if request.args.get('category'): query += " WHERE category = %s"; params.append(request.args.get('category'))
            cursor.execute(query + " ORDER BY created_at DESC LIMIT %s", params + [int(request.args.get('limit', 50))])
            cols = ['id','user_id','username','category','rating','title','body','is_anonymous','created_at','updated_at']
            result = []
            for row in cursor.fetchall():
                d = dict(zip(cols, row))
                if d['is_anonymous'] and session.get('role') != 'Admin': d['username'] = 'Anonymous'
                d['created_at'] = d['created_at'].isoformat() if d['created_at'] else None
                result.append(d)
            return jsonify(result)
            
    data = request.get_json()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reviews (user_id, username, category, rating, title, body, is_anonymous)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, created_at
        """, (session['user_id'], session['username'], data.get('category', 'General'), int(data.get('rating', 0)), data.get('title', ''), data.get('body', ''), bool(data.get('is_anonymous', False))))
        row = cursor.fetchone()
    return jsonify({'success': True, 'id': row[0], 'created_at': row[1].isoformat()}), 201

@app.route('/api/reviews/<int:review_id>', methods=['DELETE'])
@api_login_required
def delete_review(review_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM reviews WHERE id = %s", (review_id,))
        row = cursor.fetchone()
        if not row: return jsonify({'error': 'Not found'}), 404
        if row[0] != session['user_id'] and session.get('role') != 'Admin': return jsonify({'error': 'Denied'}), 403
        cursor.execute("DELETE FROM reviews WHERE id = %s", (review_id,))
    return jsonify({'success': True})

# ==========================================================
# MAP ANNOTATIONS, NOTES & TASKS API
# ==========================================================

def _compute_representative_point(shape_type, geojson_str, center_lat=None, center_lng=None):
    """
    Calculates the exact (lat, lng) center point for any shape so
    Tasks/Notes can fly to the correct map location.
    """
    try:
        if shape_type in ('circle', 'buffer') and center_lat is not None and center_lng is not None:
            return center_lat, center_lng

        geo = json.loads(geojson_str) if isinstance(geojson_str, str) else geojson_str
        if geo.get('type') == 'FeatureCollection':
            features = geo.get('features', [])
            geo = features[0].get('geometry', {}) if features else {}
        elif geo.get('type') == 'Feature':
            geo = geo.get('geometry', {})

        gtype = geo.get('type', '')
        coords = geo.get('coordinates', [])

        def flatten_coords(c):
            if not c:
                return []
            if isinstance(c[0], (int, float)):
                return [c]
            result = []
            for item in c:
                result.extend(flatten_coords(item))
            return result

        flat = flatten_coords(coords)
        if not flat:
            return None, None

        if gtype == 'Point':
            return flat[0][1], flat[0][0]

        if gtype == 'LineString':
            mid = flat[len(flat) // 2]
            return mid[1], mid[0]

        lngs = [c[0] for c in flat]
        lats = [c[1] for c in flat]
        return sum(lats) / len(lats), sum(lngs) / len(lngs)

    except Exception:
        return None, None


@app.route('/api/annotations', methods=['GET'])
@api_login_required
def get_annotations():
    try:
        status_filter = request.args.get('status', '')
        user_id = session['user_id']

        base_q = """
            SELECT DISTINCT
                a.id, a.title, a.description, a.shape_type, a.geojson,
                a.center_lat, a.center_lng, a.radius_meters,
                a.representative_lat, a.representative_lng,
                a.color, a.fill_color, a.fill_opacity, a.stroke_weight,
                a.created_by, a.created_by_username,
                a.assigned_to, a.assigned_to_username,
                a.status, a.priority,
                a.created_at, a.updated_at,
                a.closed_at, a.days_open,
                (SELECT COUNT(*) FROM annotation_comments c
                 WHERE c.annotation_id = a.id) AS comment_count
            FROM map_annotations a
            LEFT JOIN annotation_assignees aa ON aa.annotation_id = a.id
            WHERE (a.created_by = %s OR a.assigned_to = %s OR aa.user_id = %s)
        """
        params = [user_id, user_id, user_id]

        if status_filter:
            base_q += " AND a.status = %s"
            params.append(status_filter)

        base_q += " ORDER BY a.created_at DESC"

        cols = [
            'id', 'title', 'description', 'shape_type', 'geojson',
            'center_lat', 'center_lng', 'radius_meters',
            'representative_lat', 'representative_lng',
            'color', 'fill_color', 'fill_opacity', 'stroke_weight',
            'created_by', 'created_by_username',
            'assigned_to', 'assigned_to_username',
            'status', 'priority', 'created_at', 'updated_at',
            'closed_at', 'days_open', 'comment_count'
        ]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(base_q, params)
                rows = cur.fetchall()

                ann_ids = [r[0] for r in rows]
                assignees_map = {}
                if ann_ids:
                    cur.execute("""
                        SELECT aa.annotation_id, u.id, u.username, u.full_name
                        FROM annotation_assignees aa
                        JOIN users u ON u.id = aa.user_id
                        WHERE aa.annotation_id = ANY(%s)
                        ORDER BY aa.annotation_id, u.full_name
                    """, (ann_ids,))
                    for ann_id, uid, uname, fname in cur.fetchall():
                        assignees_map.setdefault(ann_id, []).append({
                            'id': uid, 'username': uname, 'full_name': fname or uname
                        })

        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d['created_at'] = d['created_at'].isoformat() if d['created_at'] else None
            d['updated_at'] = d['updated_at'].isoformat() if d['updated_at'] else None
            d['closed_at']  = d['closed_at'].isoformat()  if d['closed_at']  else None
            d['assignees'] = assignees_map.get(d['id'], [])
            if d['assignees']:
                d['assigned_to_username'] = ', '.join(a['full_name'] for a in d['assignees'])
            result.append(d)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations', methods=['POST'])
@api_login_required
def create_annotation():
    try:
        data     = request.get_json()
        user_id  = session['user_id']
        username = session['username']

        assigned_ids = data.get('assigned_to_ids') or []
        if not assigned_ids and data.get('assigned_to'):
            assigned_ids = [int(data['assigned_to'])]
        assigned_ids = [int(x) for x in assigned_ids if x]

        assigned_to          = assigned_ids[0] if assigned_ids else None
        assigned_to_username = None

        geojson = data.get('geojson')
        if isinstance(geojson, dict):
            geojson = json.dumps(geojson)

        shape_type = data.get('shape_type', 'polygon')
        rep_lat, rep_lng = _compute_representative_point(
            shape_type, geojson,
            center_lat=data.get('center_lat'),
            center_lng=data.get('center_lng')
        )

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if assigned_to:
                    cur.execute("SELECT username FROM users WHERE id = %s", (assigned_to,))
                    row = cur.fetchone()
                    assigned_to_username = row[0] if row else None

                cur.execute("""
                    INSERT INTO map_annotations
                        (title, description, shape_type, geojson,
                         center_lat, center_lng, radius_meters,
                         representative_lat, representative_lng,
                         color, fill_color, fill_opacity, stroke_weight,
                         created_by, created_by_username,
                         assigned_to, assigned_to_username,
                         status, priority)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id, created_at
                """, (
                    data.get('title', 'Untitled'),
                    data.get('description', ''),
                    shape_type,
                    geojson,
                    data.get('center_lat'),
                    data.get('center_lng'),
                    data.get('radius_meters'),
                    rep_lat,
                    rep_lng,
                    data.get('color', '#2563eb'),
                    data.get('fill_color', '#2563eb'),
                    data.get('fill_opacity', 0.2),
                    data.get('stroke_weight', 2),
                    user_id,
                    username,
                    assigned_to,
                    assigned_to_username,
                    data.get('status', 'open'),
                    data.get('priority', 'normal'),
                ))
                new_id, created_at = cur.fetchone()

                if assigned_ids:
                    for aid in assigned_ids:
                        cur.execute("""
                            INSERT INTO annotation_assignees (annotation_id, user_id)
                            VALUES (%s, %s) ON CONFLICT DO NOTHING
                        """, (new_id, aid))

        return jsonify({
            'id': new_id,
            'created_at': created_at.isoformat(),
            'representative_lat': rep_lat,
            'representative_lng': rep_lng,
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations/<int:ann_id>', methods=['PUT'])
@api_login_required
def update_annotation(ann_id):
    try:
        data = request.get_json()

        assigned_ids = data.get('assigned_to_ids') or []
        if not assigned_ids and data.get('assigned_to'):
            assigned_ids = [int(data['assigned_to'])]
        assigned_ids = [int(x) for x in assigned_ids if x]

        assigned_to          = assigned_ids[0] if assigned_ids else None
        assigned_to_username = None

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT created_by FROM map_annotations WHERE id = %s", (ann_id,))
                row = cur.fetchone()
                if not row:
                    return jsonify({'error': 'Not found'}), 404
                if row[0] != session['user_id'] and session.get('role') != 'Admin':
                    return jsonify({'error': 'Unauthorized'}), 403

                if assigned_to:
                    cur.execute("SELECT username FROM users WHERE id = %s", (assigned_to,))
                    ur = cur.fetchone()
                    assigned_to_username = ur[0] if ur else None

                cur.execute("""
                    UPDATE map_annotations SET
                        title                = %s,
                        description          = %s,
                        assigned_to          = %s,
                        assigned_to_username = %s,
                        status               = %s,
                        priority             = %s,
                        color                = %s,
                        fill_color           = %s
                    WHERE id = %s
                """, (
                    data.get('title'),
                    data.get('description'),
                    assigned_to,
                    assigned_to_username,
                    data.get('status'),
                    data.get('priority'),
                    data.get('color', '#2563eb'),
                    data.get('fill_color', '#2563eb'),
                    ann_id,
                ))

                cur.execute("DELETE FROM annotation_assignees WHERE annotation_id = %s", (ann_id,))
                for aid in assigned_ids:
                    cur.execute("""
                        INSERT INTO annotation_assignees (annotation_id, user_id)
                        VALUES (%s, %s) ON CONFLICT DO NOTHING
                    """, (ann_id, aid))

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations/<int:ann_id>', methods=['DELETE'])
@api_login_required
def delete_annotation(ann_id):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT created_by FROM map_annotations WHERE id = %s", (ann_id,))
                row = cur.fetchone()
                if not row:
                    return jsonify({'error': 'Not found'}), 404
                if row[0] != session['user_id'] and session.get('role') != 'Admin':
                    return jsonify({'error': 'Unauthorized'}), 403

                cur.execute("DELETE FROM map_annotations WHERE id = %s", (ann_id,))

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations/<int:ann_id>/comments', methods=['GET', 'POST'])
@api_login_required
def handle_annotation_comments(ann_id):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if request.method == 'GET':
                    cur.execute("""
                        SELECT id, author_id, author_username, body, created_at
                        FROM annotation_comments
                        WHERE annotation_id = %s
                        ORDER BY created_at ASC
                    """, (ann_id,))
                    rows = cur.fetchall()
                    result = [
                        {'id': r[0], 'author_id': r[1], 'author_username': r[2], 'body': r[3], 'created_at': r[4].isoformat()}
                        for r in rows
                    ]
                    return jsonify(result)

                data = request.get_json()
                cur.execute("""
                    INSERT INTO annotation_comments
                        (annotation_id, author_id, author_username, body)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, created_at
                """, (ann_id, session['user_id'], session['username'], data.get('body', '')))
                new_id, created_at = cur.fetchone()
        return jsonify({'id': new_id, 'created_at': created_at.isoformat()}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/list', methods=['GET'])
@api_login_required
def list_users_for_assign():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, role FROM users WHERE is_active = TRUE ORDER BY full_name")
        return jsonify([{'id': r[0], 'username': r[1], 'full_name': r[2], 'role': r[3]} for r in cursor.fetchall()])

def get_pricing_flat():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, action_name, price_myr, price_min, price_max FROM capex_pricing ORDER BY category, action_name;")
            flat = {}
            for category, action_name, price_myr, price_min, price_max in cursor.fetchall():
                flat.setdefault(category, {})[action_name] = {"price": float(price_myr), "min": float(price_min), "max": float(price_max)}
            return flat
    except Exception: return {}

def get_pricing_for_calc():
    flat = get_pricing_flat()
    return {cat: {name: vals["price"] for name, vals in items.items()} for cat, items in flat.items()}

def get_pricing_ranges():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, action_name, price_min, price_max FROM capex_pricing ORDER BY category, action_name;")
            ranges = {}
            for category, action_name, price_min, price_max in cursor.fetchall():
                ranges.setdefault(category, {})[action_name] = {"min": float(price_min), "max": float(price_max), "display": f"RM {float(price_min):,.2f} \u2013 RM {float(price_max):,.2f}"}
            return ranges
    except Exception: return {}

@app.route('/api/pricing', methods=['GET', 'POST'])
@api_login_required
def pricing_endpoint():
    role = session.get('role', 'Staff')
    if request.method == 'POST':
        if role not in ['Admin', 'Planner']: return jsonify({'error': 'Unauthorized'}), 403
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for category, items in request.json.items():
                    for action_name, vals in items.items():
                        cursor.execute("UPDATE capex_pricing SET price_myr=%s, price_min=%s, price_max=%s, updated_by=%s WHERE category=%s AND action_name=%s", (vals.get('price',0), vals.get('min',0), vals.get('max',0), session.get('user_id'), category, action_name))
            return jsonify({"success": True, "message": "Pricing updated successfully!"})
        except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

    if role in ['Admin', 'Planner']: return jsonify(get_pricing_flat())
    return jsonify(get_pricing_ranges())

# --- ATHENA DATA ENDPOINTS ---
@app.route('/api/years')
def api_years():
    try:
        sql = "SELECT DISTINCT year FROM sector_calculations ORDER BY year DESC"
        df = get_cached_dataframe(sql)
        return jsonify(df['year'].tolist())
    except Exception as e:
        print(f"Athena Error: {e}")
        return jsonify([datetime.now().year])

@app.route('/api/weeks')
def api_weeks():
    year = request.args.get('year', type=int)
    try:
        sql = f"SELECT DISTINCT week FROM sector_calculations WHERE year = {year} ORDER BY week DESC" if year else "SELECT DISTINCT week FROM sector_calculations ORDER BY week DESC"
        df = get_cached_dataframe(sql)
        return jsonify(df['week'].tolist())
    except Exception: return jsonify([])

@app.route('/api/filters/regions')
def api_filters_regions():
    try:
        sql = "SELECT DISTINCT UPPER(region) as reg FROM sector_calculations WHERE region IS NOT NULL ORDER BY UPPER(region)"
        df = get_cached_dataframe(sql)
        return jsonify(df['reg'].tolist())
    except Exception: return jsonify([])

@app.route('/api/superset/guest-token')
@api_login_required
def get_superset_guest_token():
    dashboard_id = request.args.get('dashboard_id')
    if not dashboard_id:
        return jsonify({"error": "Dashboard ID required"}), 400

    try:
        # 1. Authenticate with Superset internally over the Docker network
        login_res = requests.post(
            'http://superset:8088/api/v1/security/login',
            json={"username": "admin", "password": "admin", "provider": "db"}, # Replace with your actual admin password
            timeout=5
        )
        login_res.raise_for_status()
        access_token = login_res.json().get('access_token')

        # 2. Request a temporary Guest Token for the specific dashboard
        guest_token_res = requests.post(
            'http://superset:8088/api/v1/security/guest_token/',
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "user": {
                    "username": session.get('username'), 
                    "first_name": "NetAlytics", 
                    "last_name": "Admin"
                },
                "resources": [{"type": "dashboard", "id": dashboard_id}],
                "rls": [] # Row Level Security (we can use this later to filter data by region!)
            },
            timeout=5
        )
        guest_token_res.raise_for_status()
        
        return jsonify({"token": guest_token_res.json().get('token')})

    except Exception as e:
        print(f"Superset Token Error: {e}")
        return jsonify({"error": "Failed to communicate with analytics engine"}), 500

@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    try:
        year = request.args.get('year', str(datetime.now().year))

        # 1. Fetch Global RAM Cache
        sql_sc = f"SELECT zoom_sector_id, region, operator, cluster, week, eric_data_volume_ul_dl FROM sector_calculations WHERE CAST(year AS VARCHAR) = '{year}'"
        df_sc = get_cached_dataframe(sql_sc)

        sql_ca = f"SELECT zoom_sector_id, region, operator, cluster, week, congested FROM congestion_analysis WHERE CAST(year AS VARCHAR) = '{year}'"
        df_ca = get_cached_dataframe(sql_ca)

        # 2. Instantly filter in Pandas
        df_sc_filtered = apply_pandas_filters(df_sc, request.args)
        df_ca_filtered = apply_pandas_filters(df_ca, request.args)
        df_ca_filtered = df_ca_filtered[df_ca_filtered['congested'] == True]

        # 3. Compute stats
        total_sectors = df_sc_filtered['zoom_sector_id'].str.split('_').str[0].nunique() if not df_sc_filtered.empty else 0
        avg_vol = df_sc_filtered['eric_data_volume_ul_dl'].mean() if not df_sc_filtered.empty else 0.0
        congested_count = df_ca_filtered['zoom_sector_id'].nunique() if not df_ca_filtered.empty else 0

        return jsonify({
            'total_sectors': int(total_sectors),
            'congested_count': int(congested_count),
            'avg_volume': float(avg_vol) if pd.notna(avg_vol) else 0.0
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

# --- FETCH ONCE ARCHITECTURE (No Pagination in SQL) ---
@app.route('/api/sector_data')
def api_sector_data():
    try:
        year = request.args.get('year', str(datetime.now().year))
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 25))

        # 1. Fetch Global RAM Cache
        sql = f"""
            SELECT zoom_sector_id, week, year, region, cluster,
                   ibc_macro, f1f2f3, eric_prb_util_rate, eric_dl_user_ip_thpt,
                   eric_data_volume_ul_dl, dataset_type, operator, area_target
            FROM sector_calculations
            WHERE CAST(year AS VARCHAR) = '{year}'
        """
        df = get_cached_dataframe(sql)

        # 2. Instantly filter and sort in Pandas
        df_filtered = apply_pandas_filters(df, request.args)
        df_filtered = df_filtered.sort_values(by=['zoom_sector_id', 'week'], ascending=[True, False])

        # 3. Slice for DataTables
        total_records = len(df_filtered)
        df_page = df_filtered.iloc[start : start + length]

        return jsonify({
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': total_records,
            'recordsFiltered': total_records,
            'data': df_page.replace({np.nan: None}).to_dict('records')
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/forecast_data')
def api_forecast_data():
    try:
        year = request.args.get('year', str(datetime.now().year))
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 25))

        # 1. Fetch Global RAM Caches
        sql_sc = f"""
            SELECT zoom_sector_id, CAST(week AS INTEGER) as week, CAST(year AS INTEGER) as year,
                   CAST(week AS INTEGER) / 4 + 1 as month, ibc_macro, dataset_type, operator,
                   region, cluster, 
                   CAST(eric_data_volume_ul_dl AS VARCHAR) as actual_data_volume,
                   CAST(eric_prb_util_rate AS VARCHAR) as actual_prb_util_rate,
                   CAST(eric_dl_user_ip_thpt AS VARCHAR) as actual_dl_user_ip_thpt,
                   CAST(NULL AS VARCHAR) as predicted_eric_data_volume_ul_dl,
                   CAST(NULL AS VARCHAR) as predicted_eric_prb_util_rate,
                   CAST(NULL AS VARCHAR) as predicted_eric_dl_user_ip_thpt,
                   FALSE as congested
            FROM sector_calculations
            WHERE CAST(year AS VARCHAR) = '{year}'
        """
        sql_fr = f"""
            SELECT zoom_sector_id, CAST(week AS INTEGER) as week, CAST(year AS INTEGER) as year,
                   CAST(month AS INTEGER) as month, ibc_macro, dataset_type, operator,
                   CAST(NULL AS VARCHAR) as actual_data_volume,
                   CAST(NULL AS VARCHAR) as actual_prb_util_rate,
                   CAST(NULL AS VARCHAR) as actual_dl_user_ip_thpt,
                   CAST(ROUND(predicted_eric_data_volume_ul_dl, 2) AS VARCHAR) as predicted_eric_data_volume_ul_dl,
                   CAST(ROUND(predicted_eric_prb_util_rate, 2) AS VARCHAR) as predicted_eric_prb_util_rate,
                   CAST(ROUND(predicted_eric_dl_user_ip_thpt, 2) AS VARCHAR) as predicted_eric_dl_user_ip_thpt,
                   congested
            FROM forecast_results
            WHERE CAST(year AS VARCHAR) = '{year}'
              AND CAST(week AS VARCHAR) IN ('13', '26', '39', '52')
        """
        
        df_sc = get_cached_dataframe(sql_sc)
        df_fr = get_cached_dataframe(sql_fr)

        # 2. Filter SC Data in Pandas (ignore week to keep timeline intact)
        req_args = request.args.to_dict()
        req_args.pop('week', None)
        df_sc_filtered = apply_pandas_filters(df_sc, req_args)

        # 3. Only keep Forecasts for the sectors that survived the SC filter
        valid_sectors = df_sc_filtered['zoom_sector_id'].unique()
        df_fr_filtered = df_fr[df_fr['zoom_sector_id'].isin(valid_sectors)]

        # Combine, sort, and slice
        df_combined = pd.concat([df_sc_filtered, df_fr_filtered], ignore_index=True)
        df_combined = df_combined.sort_values(by=['zoom_sector_id', 'year', 'week'], ascending=[True, True, True])

        total_records = len(df_combined)
        df_page = df_combined.iloc[start : start + length]

        return jsonify({
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': total_records,
            'recordsFiltered': total_records,
            'data': df_page.replace({np.nan: None}).to_dict('records')
        })
    except Exception as e: return jsonify({'error': str(e)}), 500


@app.route('/api/congestion_data')
def api_congestion_data():
    try:
        year = request.args.get('year', str(datetime.now().year))
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 25))

        sql = f"""
            SELECT zoom_sector_id, week, year, month, region, cluster,
                   eric_data_volume_ul_dl, eric_prb_util_rate, eric_dl_user_ip_thpt,
                   eric_max_rrc_user, max_active_user, congested_weeks, congested_count_month,
                   dataset_type, operator, area_target, bau_nic, congested
            FROM congestion_analysis
            WHERE CAST(year AS VARCHAR) = '{year}'
        """
        df = get_cached_dataframe(sql)

        # Filter in Pandas
        df_filtered = apply_pandas_filters(df, request.args)
        df_filtered = df_filtered[df_filtered['congested'] == True]
        df_filtered = df_filtered.sort_values(by=['congested_weeks', 'eric_prb_util_rate'], ascending=[False, False])

        total_records = len(df_filtered)
        df_page = df_filtered.iloc[start : start + length]

        return jsonify({
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': total_records,
            'recordsFiltered': total_records,
            'data': df_page.replace({np.nan: None}).to_dict('records')
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/sites')
def api_sites():
    selected_week = request.args.get('week')
    year = request.args.get('year', str(datetime.now().year))

    try:
        sql_coords = "SELECT site_id, region, cluster, latitude, longitude FROM site_coordinates WHERE latitude IS NOT NULL"
        df_coords = get_cached_dataframe(sql_coords)

        sql_cov = "SELECT site_id, cell_name as sector_id, azimuth, 65 as beamwidth, coverage_radius_m as radius, technology, 'Unknown' as band FROM site_coverage_params"
        df_cov = get_cached_dataframe(sql_cov)

        sql_cong = f"""
            SELECT
                split_part(ca.zoom_sector_id, '_', 1) as site_id, ca.region, ca.cluster, ca.week,
                ca.zoom_sector_id, ca.eric_prb_util_rate, ca.eric_dl_user_ip_thpt, ca.eric_data_volume_ul_dl,
                GREATEST(COALESCE(ca.eric_max_rrc_user,0), COALESCE(ca.max_active_user,0)) as users,
                ca.congested_weeks, ca.month, ca.congested_count_month, ca.operator, ca.area_target, ca.bau_nic,
                cu.current_f1_l9, cu.current_f1_l18, cu.current_f1_l21, cu.current_f1_l26,
                cu.current_f2_l9, cu.current_f2_l18, cu.current_f2_l21, cu.current_f2_l26
            FROM congestion_analysis ca
            LEFT JOIN capex_upgrades cu
                ON TRIM(UPPER(ca.zoom_sector_id)) = TRIM(UPPER(cu.zoom_sector_id))
                AND CAST(ca.year AS VARCHAR) = CAST(cu.data_year AS VARCHAR)
                AND CAST(ca.week AS VARCHAR) = CAST(cu.data_week AS VARCHAR)
            WHERE CAST(ca.year AS VARCHAR) = '{year}'
        """
        df_cong = get_cached_dataframe(sql_cong)

        # 1. First Pandas Filter (Region, Operator, Cluster)
        df_cong_filtered = apply_pandas_filters(df_cong, request.args)

        # 2. Strict Week Filter logic
        if not selected_week or str(selected_week).lower() == 'all':
            selected_week = int(df_cong_filtered['week'].max()) if not df_cong_filtered.empty else 1
        else:
            selected_week = int(selected_week)
            
        df_cong_filtered = df_cong_filtered[df_cong_filtered['week'].astype(int) == selected_week]

        coords_list = df_coords.to_dict('records')
        cov_list = df_cov.to_dict('records')
        cong_list = df_cong_filtered.to_dict('records')

        sites_map = {}
        for row in coords_list:
            sid = str(row['site_id']).upper()
            sites_map[sid] = {
                'site_id': sid, 'region': row['region'], 'cluster': row['cluster'],
                'lat': float(row['latitude']) if pd.notna(row['latitude']) else 0.0,
                'lng': float(row['longitude']) if pd.notna(row['longitude']) else 0.0,
                'sectors': [], 'coverage': [], 'max_cong_weeks': 0, 'data_week': selected_week,
                'area_target': 'Unknown', 'bau_nic': 'Unknown', 'operator': 'Unknown', 'band_matrix': []
            }

        for row in cov_list:
            sid = str(row['site_id']).upper()
            if sid in sites_map:
                sites_map[sid]['coverage'].append({
                    'sec': row['sector_id'], 'az': float(row['azimuth']) if pd.notna(row['azimuth']) else 0.0,
                    'bw': float(row['beamwidth']) if pd.notna(row['beamwidth']) else 65.0,
                    'rad': float(row['radius']) if pd.notna(row['radius']) else 1000.0,
                    'tech': row['technology'], 'band': row['band']
                })

        for row in cong_list:
            sid = str(row['site_id']).upper()
            if sid in sites_map:
                sites_map[sid]['sectors'].append({
                    'name': row['zoom_sector_id'], 'prb': row['eric_prb_util_rate'],
                    'thpt': row['eric_dl_user_ip_thpt'], 'vol': row['eric_data_volume_ul_dl'],
                    'users': row['users'], 'month': row['month'], 'cong_month_cnt': row['congested_count_month']
                })
                cw = row['congested_weeks']
                sites_map[sid]['max_cong_weeks'] = max(sites_map[sid]['max_cong_weeks'], cw if pd.notna(cw) else 0)
                sites_map[sid]['operator'] = row['operator']
                sites_map[sid]['area_target'] = row['area_target']
                sites_map[sid]['bau_nic'] = row['bau_nic']

                for c in ['f1', 'f2']:
                    for b in ['l9', 'l18', 'l21', 'l26']:
                        val = row.get(f"current_{c}_{b}")
                        if pd.notna(val) and str(val).strip() not in ["0", "", "nan", "None"]:
                            sites_map[sid]['band_matrix'].append({
                                'sector': row['zoom_sector_id'],
                                'f1f2f3': c.upper(), 'band': b.upper(), 'xtxr': str(val).strip()
                            })

        active_sites = [site for site in sites_map.values() if len(site['sectors']) > 0]
        return jsonify(active_sites)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/map/holes')
def get_map_holes():
    try:
        sql = "SELECT latitude, longitude, signal_strength, cluster_id, serving_cell, data_source FROM coverage_holes_clustered LIMIT 10000"
        df = get_cached_dataframe(sql)
        features = [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r['longitude'], r['latitude']]},
            "properties": {"signal": r['signal_strength'], "cluster": r['cluster_id'], "serving_cell": r['serving_cell'], "data_source": r['data_source']}
        } for _, r in df.iterrows()]
        return jsonify({"type": "FeatureCollection", "features": features})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/site_ids')
def api_site_ids():
    q = request.args.get('q', '').upper().strip()
    if len(q) < 2: return jsonify([])
    try:
        # Fetch from RAM instead of triggering Athena
        sql = f"SELECT DISTINCT site_id FROM site_coordinates WHERE UPPER(site_id) LIKE '%{q}%' LIMIT 10"
        df = get_cached_dataframe(sql)
        return jsonify(df['site_id'].tolist())
    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify([])

@app.route('/api/map/top_congested')
def api_map_top_congested():
    try:
        year = request.args.get('year', str(datetime.now().year))
        week = request.args.get('week', type=int) or 40

        sql = f"""
            SELECT zoom_sector_id, eric_prb_util_rate as prb, congested_weeks, region, week, congested
            FROM congestion_analysis
            WHERE CAST(year AS VARCHAR) = '{year}'
        """
        df = get_cached_dataframe(sql)

        # Filter in Pandas
        df_filtered = apply_pandas_filters(df, request.args)
        df_filtered = df_filtered[(df_filtered['congested'] == True) & (df_filtered['week'].astype(int) == week)]
        df_filtered = df_filtered.sort_values(by=['congested_weeks', 'prb'], ascending=[False, False]).head(10)

        return jsonify([{
            "zoom_sector_id": r['zoom_sector_id'],
            "congested_weeks": int(r['congested_weeks']) if pd.notna(r['congested_weeks']) else 0,
            "prb": round(float(r['prb']), 2) if pd.notna(r['prb']) else 0.0
        } for _, r in df_filtered.iterrows()])
    except Exception as e:
        print(f"Leaderboard Error: {e}")
        return jsonify([])


@app.route('/api/map/worst_clusters')
def api_map_worst_clusters():
    try:
        sql_mr = """
            SELECT cluster_id, COUNT(*) as point_count, AVG(signal_strength) as avg_signal,
                   AVG(latitude) as center_lat, AVG(longitude) as center_lon
            FROM coverage_holes_clustered
            WHERE data_source = 'MR' AND cluster_id != -1
            GROUP BY cluster_id
            ORDER BY point_count DESC LIMIT 10
        """
        df_mr = get_cached_dataframe(sql_mr)

        sql_ookla = """
            SELECT cluster_id, COUNT(*) as point_count, AVG(signal_strength) as avg_signal,
                   AVG(latitude) as center_lat, AVG(longitude) as center_lon
            FROM coverage_holes_clustered
            WHERE data_source = 'Ookla' AND cluster_id != -1
            GROUP BY cluster_id
            ORDER BY point_count DESC LIMIT 10
        """
        df_ookla = get_cached_dataframe(sql_ookla)

        return jsonify({
            "mr": df_mr.replace({np.nan: None}).to_dict('records') if not df_mr.empty else [],
            "ookla": df_ookla.replace({np.nan: None}).to_dict('records') if not df_ookla.empty else []
        })
    except Exception as e:
        print(f"Worst Clusters Error: {e}")
        return jsonify({"mr": [], "ookla": []})

# --- INTERACTIVE FORECAST PLOTTING ---
@app.route('/plot')
def plot_route():
    site_id = request.args.get('site_id')
    forecast_horizon = request.args.get('forecast_horizon', default=52, type=int)
    if not site_id: return jsonify({'error': 'Missing site_id'}), 400

    try:
        METRICS = [
            {'col': 'eric_data_volume_ul_dl', 'title': 'Data Volume (GB)',  'color': '#1f77b4', 'limit': None},
            {'col': 'eric_prb_util_rate',     'title': 'PRB Util (%)',      'color': '#ff7f0e', 'limit': 100},
            {'col': 'eric_dl_user_ip_thpt',   'title': 'Throughput (Mbps)', 'color': '#2ca02c', 'limit': None}
        ]

        sql = f"""
            SELECT zoom_sector_id, week, year, eric_data_volume_ul_dl, eric_prb_util_rate, eric_dl_user_ip_thpt
            FROM sector_calculations WHERE zoom_sector_id LIKE '{site_id.strip()}%' ORDER BY year, week
        """
        df_actual = get_cached_dataframe(sql)
        
        if df_actual.empty: return jsonify({'error': 'No data found'}), 404

        def get_date(r):
            try: return date.fromisocalendar(int(r['year']), int(r['week']), 1)
            except: return None

        df_actual['plot_date'] = pd.to_datetime(df_actual.apply(get_date, axis=1))
        df_actual = df_actual.dropna(subset=['plot_date'])
        start_date = df_actual['plot_date'].min()

        all_plots = []
        sectors = df_actual['zoom_sector_id'].unique()

        for i, sector in enumerate(sectors):
            df_sec = df_actual[df_actual['zoom_sector_id'] == sector].sort_values('plot_date')
            df_sec['days'] = (df_sec['plot_date'] - start_date).dt.days
            x_raw = df_sec['days'].values.reshape(-1, 1)
            last_day = x_raw.max()
            future_days_col = np.arange(last_day + 7, last_day + (7 * forecast_horizon), 7).reshape(-1, 1)
            future_dates = [start_date + timedelta(days=int(d)) for d in future_days_col.flatten()]

            row_plots = []
            for j, metric in enumerate(METRICS):
                y_raw = df_sec[metric['col']].values
                mask = ~np.isnan(y_raw)
                p = figure(title=f"{sector} - {metric['title']}" if j==1 else (sector if j==0 else metric['title']), x_axis_type="datetime", sizing_mode="stretch_width", height=280, tools="pan,wheel_zoom,reset,save", background_fill_color="#fafafa")

                if np.sum(mask) > 2:
                    x_clean = x_raw[mask]; y_clean = y_raw[mask]; n = len(x_clean)
                    model = LinearRegression(); model.fit(x_clean, y_clean)
                    y_pred = model.predict(future_days_col)
                    x_mean = np.mean(x_clean); y_hat_hist = model.predict(x_clean)
                    residuals = y_clean - y_hat_hist; rss = np.sum(residuals**2); dof = n - 2
                    s_err = np.sqrt(rss / dof); sxx = np.sum((x_clean - x_mean)**2)
                    t_val = t_dist.ppf(0.975, dof)

                    ci_width = [t_val * (s_err * np.sqrt((1/n) + ((d - x_mean)**2 / sxx))) for d in future_days_col.flatten()]
                    y_pred = np.maximum(y_pred, 0)
                    if metric['limit']: y_pred = np.minimum(y_pred, metric['limit'])
                    upper = y_pred + ci_width; lower = np.maximum(y_pred - ci_width, 0)
                    if metric['limit']: upper = np.minimum(upper, metric['limit'])

                    band_x = np.append(future_dates, future_dates[::-1]); band_y = np.append(lower, upper[::-1])
                    p.patch(band_x, band_y, color=metric['color'], alpha=0.15, line_width=0)
                    p.line(future_dates, y_pred, color=metric['color'], line_dash="dashed", line_width=1.5, legend_label="Forecast")

                    source_actual = ColumnDataSource(data=dict(date=df_sec['plot_date'], val=df_sec[metric['col']], week_num=df_sec['week']))
                    p.line('date', 'val', source=source_actual, color=metric['color'], line_width=1.5, legend_label="Actual")
                    c = p.scatter('date', 'val', source=source_actual, color=metric['color'], size=5, marker="circle")
                    p.add_tools(HoverTool(renderers=[c], tooltips=[("Week", "@week_num"), ("Val", "@val{0.2f}")], formatters={'@date': 'datetime'}))

                p.legend.location = "top_left"; p.legend.label_text_font_size = "7pt"
                row_plots.append(p)
            all_plots.append(row_plots)

        grid = gridplot(all_plots, toolbar_location="right", sizing_mode="stretch_width")
        return jsonify({'plot_image': json.dumps(json_item(grid, "myplot"))})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# --- ADMIN PRICING LOGIC (KEPT INTACT) ---
DEFAULT_PRICING = {
    "EQ": { "BW Upg": 2500.00,
            "Add Layer": 38877.25,
            "Bi-Sect Radio": 47119.72,
            "Bi-Sect Antenna + Accessory": 6000.00,
            "MM": 83637.42,
            "Swap all Sector Radio Ericsson to ZTE": 120000.00,
            "Add Sector Outdoor": 40000.00,
            "Add Sector IBC": 10000.00,
            "Accelerate NIC": 62000.00,
            "NNS": 250000.00,
            "Split Omni to Sector": 116631.75
            },
    "ES": { "BW Upg": 1350.00,
            "Add Layer": 32000.00,
            "Bi-Sect": 34000.00,
            "MM": 36000.00,
            "Dismantle": 25000.00,
            "Split Omni to Sector": 40000.00,
            "Swap all sector radio Ericsson to ZTE": 40000.00,
            "Add Sector Outdoor": 13000.00,
            "Add Sector IBC": 13000.00,
            "Accelerate NIC": 24000.00,
            "NNS": 40000.00
            }
}

def get_pricing():
    # Try to fetch the absolute latest pricing from S3 so the Dashboard always matches AWS Glue
    try:
        s3_client = aws_session.client('s3')
        response = s3_client.get_object(Bucket='neo-advanced-analytics', Key='capex_pricing/capex_pricing.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Could not read pricing from S3, falling back to local/default. Error: {e}")
        if os.path.exists(PRICING_FILE):
            with open(PRICING_FILE, 'r') as f: return json.load(f)
        return DEFAULT_PRICING

@app.route('/api/pricing', methods=['GET', 'POST'])
def handle_pricing():
    if request.method == 'POST':
        new_pricing = request.json
        
        # 1. Save Locally (as a backup cache)
        with open(PRICING_FILE, 'w') as f: 
            json.dump(new_pricing, f, indent=4)
            
        # 2. Push to S3 so AWS Glue uses the new prices on its next run!
        try:
            s3_client = aws_session.client('s3')
            s3_client.put_object(
                Bucket='neo-advanced-analytics',
                Key='capex_pricing/capex_pricing.json',
                Body=json.dumps(new_pricing, indent=4),
                ContentType='application/json'
            )
            return jsonify({"success": True, "message": "Pricing updated successfully and pushed to AWS S3!"})
        except Exception as e:
            print(f"Error uploading pricing to S3: {e}")
            return jsonify({"success": False, "message": f"Saved locally, but failed to sync with AWS: {str(e)}"}), 500

    return jsonify(get_pricing())

def recalculate_live_capex(row, pricing):
    case_str = str(row.get('suggested_upgrade_case', ''))
    if not case_str or case_str.lower() in ['nan', 'none', '']:
        return 0.0, 0.0, 0.0

    # Dynamically count how many layers were added by comparing Current vs Suggested
    added_layers = 0
    for c in ['f1', 'f2']:
        for b in ['l9', 'l18', 'l21', 'l26']:
            curr = str(row.get(f'current_{c}_{b}', '0')).strip().lower()
            sugg = str(row.get(f'suggested_{c}_{b}', '0')).strip().lower()
            if curr in ['0', '', 'none', 'nan', '<na>'] and sugg not in ['0', '', 'none', 'nan', '<na>']:
                added_layers += 1

    eq_prices = pricing.get("EQ", {})
    es_prices = pricing.get("ES", {})

    eq_costs = []
    es_options = []

    # Apply the Engineering layer multiplier
    layer_mult = {1: 1.0, 2: 1.7, 3: 2.7, 4: 3.5, 5: 4.5, 6: 5.5, 7: 6.5, 8: 7.2}.get(added_layers, 1.0) if added_layers > 0 else 0
    add_layer_eq_cost = eq_prices.get("Add Layer", 0) * layer_mult

    case_lower = case_str.lower()

    # Base Cases
    if "case 11" in case_lower:
        eq_costs.append(eq_prices.get("NNS", 0))
        es_options.append(es_prices.get("NNS", 0))
    elif "case 4" in case_lower:
        eq_costs.append(eq_prices.get("MM", 0))
        es_options.append(es_prices.get("MM", 0))
    else:
        if "bandwidth" in case_lower or "case 1 " in case_lower:
            eq_costs.append(eq_prices.get("BW Upg", 0))
            es_options.append(es_prices.get("BW Upg", 0))
        if "layer" in case_lower or "case 3 " in case_lower:
            eq_costs.append(add_layer_eq_cost)
            es_options.append(es_prices.get("Add Layer", 0))
        if "bi-sect" in case_lower or "case 2 " in case_lower:
            eq_costs.extend([eq_prices.get("Bi-Sect Radio", 0), eq_prices.get("Bi-Sect Antenna + Accessory", 0)])
            es_options.append(es_prices.get("Bi-Sect", 0))

    # Add-ons
    if "case 8" in case_lower:
        eq_costs.append(eq_prices.get("Add Sector IBC", 0))
        es_options.append(es_prices.get("Add Sector IBC", 0))
    if "case 9" in case_lower:
        eq_costs.extend([eq_prices.get("Bi-Sect Radio", 0), eq_prices.get("Bi-Sect Antenna + Accessory", 0)])
        es_options.append(es_prices.get("Bi-Sect", 0))
    if "case 10" in case_lower:
        eq_costs.append(eq_prices.get("Accelerate NIC", 0))
        es_options.append(es_prices.get("Accelerate NIC", 0))
    if "case 12" in case_lower:
        eq_costs.append(eq_prices.get("Swap all Sector Radio Ericsson to ZTE", 0))
        es_options.append(es_prices.get("Swap all sector radio Ericsson to ZTE", 0))

    final_eq = sum(eq_costs)
    final_es = max(es_options) if es_options else 0
    return final_eq + final_es, final_eq, final_es

# --- KEEPING YOUR UPGRADE CALCULATION IN EC2 ---
@app.route('/api/map/site_upgrade_details')
def api_site_upgrade_details():
    site_id = request.args.get('site_id')
    week = request.args.get('week')
    year = request.args.get('year', str(datetime.now().year))
    
    if not site_id: return jsonify({'error': 'No Site ID'}), 400
    if not week or week.lower() == 'all': week = 40

    try:
        sql = f"""
            SELECT
                ca.zoom_sector_id,
                ca.eric_prb_util_rate,
                ca.area_target as sc_area_target,
                cu.suggested_upgrade_case,
                cu.estimated_total_capex_rm,
                cu.eq_capex_rm,
                cu.es_capex_rm,
                cu.projected_prb_pct,
                cu.current_f1_l9, cu.suggested_f1_l9,
                cu.current_f1_l18, cu.suggested_f1_l18,
                cu.current_f1_l21, cu.suggested_f1_l21,
                cu.current_f1_l26, cu.suggested_f1_l26,
                cu.current_f2_l9, cu.suggested_f2_l9,
                cu.current_f2_l18, cu.suggested_f2_l18,
                cu.current_f2_l21, cu.suggested_f2_l21,
                cu.current_f2_l26, cu.suggested_f2_l26
            FROM congestion_analysis ca
            LEFT JOIN capex_upgrades cu
                ON TRIM(UPPER(ca.zoom_sector_id)) = TRIM(UPPER(cu.zoom_sector_id))
                AND CAST(ca.year AS VARCHAR) = CAST(cu.data_year AS VARCHAR)
                AND CAST(ca.week AS VARCHAR) = CAST(cu.data_week AS VARCHAR)
            WHERE split_part(ca.zoom_sector_id, '_', 1) = '{site_id}'
            AND CAST(ca.year AS VARCHAR) = '{year}'
            AND CAST(ca.week AS VARCHAR) = '{week}'
        """
        df = get_cached_dataframe(sql)
        
        if df.empty:
            return jsonify({"error": "No sector data found for this week."})

        area_tgt = df['sc_area_target'].iloc[0] if pd.notna(df['sc_area_target'].iloc[0]) else 'Unknown'
        sectors_dict = {}
        
        # FETCH THE LIVE PRICING FROM YOUR ADMIN PANEL
        live_pricing = get_pricing()

        for _, row in df.iterrows():
            sec_id = row['zoom_sector_id']
            prb = float(row['eric_prb_util_rate']) if pd.notna(row['eric_prb_util_rate']) else 0.0

            area_str = str(row.get('sc_area_target', '')).lower()
            is_urban = 'urban' in area_str or 'kmc' in area_str
            prb_threshold = 80.0 if is_urban else 92.0

            suggested_case_str = str(row['suggested_upgrade_case']).strip()
            has_upgrade = pd.notna(row['suggested_upgrade_case']) and suggested_case_str.lower() not in ['nan', 'none', '']

            matrix = { "F1": {}, "F2": {}, "F3": {} }
            bands = ['L9', 'L18', 'L21', 'L26']
            carriers = ['F1', 'F2', 'F3']

            for c in carriers:
                for b in bands:
                    matrix[c][b] = {"curr": "-", "sugg": "-"}

            capex_rm = 0.0
            eq_cost = 0.0
            es_cost = 0.0

            if has_upgrade:
                case_label = suggested_case_str
                
                # --- LIVE RECALCULATION: Override AWS database with your Live Admin Prices ---
                live_total, live_eq, live_es = recalculate_live_capex(row, live_pricing)
                
                capex_rm = live_total
                eq_cost = live_eq
                es_cost = live_es
                
                proj_prb = float(row['projected_prb_pct']) if pd.notna(row['projected_prb_pct']) else prb

                for c in carriers:
                    for b in bands:
                        col_curr = f"current_{c.lower()}_{b.lower()}"
                        col_sugg = f"suggested_{c.lower()}_{b.lower()}"

                        c_val = str(row.get(col_curr, "0")).strip()
                        s_val = str(row.get(col_sugg, "0")).strip()

                        if c_val.lower() not in ["0", "0.0", "none", "nan", "", "<na>"]:
                            matrix[c][b]["curr"] = c_val
                        if s_val.lower() not in ["0", "0.0", "none", "nan", "", "<na>"]:
                            matrix[c][b]["sugg"] = s_val
            else:
                proj_prb = prb
                if prb >= prb_threshold:
                    case_label = "MISSING FROM REFERENCE DATA"
                else:
                    case_label = "No Upgrade Needed"

            capex_data = None
            if has_upgrade and capex_rm > 0:
                capex_data = {
                    "total_capex": capex_rm,
                    "eq_breakdown": [[case_label[:45] + "...", eq_cost]],
                    "es_chosen": {"name": "Engineering Services (Highest)", "cost": es_cost}
                }

            sectors_dict[sec_id] = {
                "is_congested": has_upgrade or (prb >= prb_threshold),
                "capacity_pct": round(proj_prb, 2),
                "case_label": case_label,
                "matrix": matrix,
                "capex": capex_data
            }

        return jsonify({
            "site_id": site_id,
            "area_target": area_tgt,
            "sectors": sectors_dict
        })
    except Exception as e:
        print(f"DEBUG: Internal Error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/cd_file')
def download_cd_file():
    try:
        s3_client = aws_session.client('s3')
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'neo-advanced-analytics', 'Key': 'processed_network_data/cd-combined-results/CD_Combined_Results.csv'},
            ExpiresIn=3600
        )
        return redirect(presigned_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/sector')
def download_sector():
    try:
        s3_client = aws_session.client('s3')
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'neo-advanced-analytics', 'Key': 'processed_network_data/cd-combined-results/Sector_Metrics.csv'},
            ExpiresIn=3600
        )
        return redirect(presigned_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/congested')
def download_congested():
    try:
        s3_client = aws_session.client('s3')
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'neo-advanced-analytics', 'Key': 'processed_network_data/cd-combined-results/Congested_Sectors.csv'},
            ExpiresIn=3600
        )
        return redirect(presigned_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    return jsonify({"reply": "The VIBE AI Agent is temporarily offline for architectural upgrades.", "status": "success"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
