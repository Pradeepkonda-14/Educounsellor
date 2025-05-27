# IIT Seat Allocation Predictor

A Streamlit-based web application that helps students predict and find suitable IIT colleges based on their JEE rank, quota, gender, and seat type preferences.

## Features

- **College Finder**: Find suitable IIT colleges based on your JEE rank
- **Advanced Filtering**: Filter colleges by:
  - Quota (OPEN, OBC-NCL, SC, ST, EWS)
  - Gender (Male/Female)
  - Seat Type (Gender-Neutral/Female-Only)
- **Visual Analytics**: 
  - Opening Ranks Comparison
  - Program Distribution
  - Seat Type Distribution
- **Historical Data**: Uses historical JEE rank data from 2016-2024

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-seat-allotment
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

3. Use the application:
   - Enter your JEE rank
   - Select your quota category
   - Choose your gender
   - Select seat type preference
   - Click "Find Colleges" to see recommendations

## Data Structure

The application uses a CSV file (`JEE_Rank_2016_2024.csv`) containing historical JEE rank data with the following columns:
- Institute
- Academic_Program_Name
- Opening_Rank
- Closing_Rank
- Quota
- Gender
- Seat_Type

## Project Structure

```
ml-seat-allotment/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── JEE_Rank_2016_2024.csv  # Historical rank data
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sourced from historical JEE rank data
- Built with Streamlit
- Uses pandas for data manipulation
- Plotly for visualizations 