from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from flask_cors import CORS
import random
import string
# import make_image

app = Flask(__name__)
CORS(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'mini_project'
app.config['CORS_HEADERS'] = 'Content-Type'
mysql = MySQL(app)

def generate_random_alphanumeric():
    characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
    random_sequence = ''.join(random.choice(characters) for _ in range(7))

    return random_sequence



@app.route("/login", methods=['POST'])
async def login():

    try:

        data = request.get_json()

        # Access parameters from the JSON data
        email = data.get('email')
        password = data.get('password')

        cur = mysql.connection.cursor()
        users = cur.execute("SELECT * FROM CLIENT_CREDENTIALS WHERE email = %s", (email,))
        if users < 1 :
            return jsonify({"error" : True, "message" : "Invalid username"})
        else : 
            users = cur.execute("SELECT * FROM CLIENT_CREDENTIALS WHERE email = %s AND password = %s", (email,password))
            if users < 1:
                return jsonify({"error" : True, "message" : "Invalid password"})
            return jsonify({"error" : False, "message" : "Login Successful", "response" : cur.fetchall()})

    except Exception as e:
        # Handle exceptions, e.g., invalid JSON format
        return jsonify({'error': str(e)})

@app.route("/signup", methods=['POST'])
async def signup():

    try:

        data = request.get_json()

        # Access parameters from the JSON data
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        retype_password = data.get('retype_password')
        company_name = data.get('company_name')

        cur = mysql.connection.cursor()

        if password == retype_password:

            users = cur.execute("SELECT * FROM CLIENT_CREDENTIALS WHERE email = %s", (email,))
            if users >= 1 :
                return jsonify({"error" : True, "message" : "Email exists"})
            else : 
                cur.execute("SELECT COUNT(*) FROM CLIENT_CREDENTIALS")

                count = cur.fetchone()[0] + 1
                secret = generate_random_alphanumeric()
                cur.execute("INSERT INTO CLIENT_CREDENTIALS VALUES (%s,%s,%s,%s,%s);", (count,email,password,company_name,secret))
                mysql.connection.commit()
                return jsonify({"error" : False, "message" : "Signin Successful", "secret" : secret})

        return jsonify({"error" : True, "message" : "Passwords do not match"})


    except Exception as e:
        # Handle exceptions, e.g., invalid JSON format
        return jsonify({'error': str(e)})

@app.route("/all", methods=['GET'])
async def all():

    try:


        cur = mysql.connection.cursor()

        all_companies = cur.execute("SELECT company_name,uid FROM CLIENT_CREDENTIALS")

        return jsonify({"error" : False, "response" : cur.fetchall()})


    except Exception as e:
        # Handle exceptions, e.g., invalid JSON format
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
