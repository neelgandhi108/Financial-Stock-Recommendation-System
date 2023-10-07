from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model using pickle
with open('portfolio.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_stocks():
    try:
        # Get user input (date and investment capacity)
        data = request.get_json()
        input_date = data['date']
        investment_capacity = data['investment_capacity']

        # Use your model to make stock recommendations based on input
        recommended_stocks = model.recommend(input_date, investment_capacity)

        return jsonify({'Recommended Stocks': recommended_stocks})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
