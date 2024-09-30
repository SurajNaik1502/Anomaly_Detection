from flask import Flask, jsonify
from darts.datasets import ETTh2Dataset
from darts.ad import KMeansScorer, QuantileDetector
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/anomaly_detection', methods=['GET'])
def anomaly_detection():

    series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]

    train, val = series.split_before(0.6)

    scorer = KMeansScorer(k=2, window=5)
    scorer.fit(train)

    anom_score = scorer.score(val)

    detector = QuantileDetector(high_quantile=0.99)
    detector.fit(scorer.score(train))

    binary_anom = detector.detect(anom_score)


    anomaly_dict = {
        "time": [str(time) for time in val.time_index],
        "anomaly_score": anom_score.values().tolist(),
        "binary_anomaly": binary_anom.values().tolist()
    }

    return jsonify(anomaly_dict)

@app.route('/plot', methods=['GET'])
def plot_anomalies():
    series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]

    train, val = series.split_before(0.6)

    scorer = KMeansScorer(k=2, window=5)
    scorer.fit(train)

    anom_score = scorer.score(val)
    
    detector = QuantileDetector(high_quantile=0.99)
    detector.fit(scorer.score(train))

    binary_anom = detector.detect(anom_score)
    series.plot()
    
    (anom_score / 2. - 100).plot(label="computed anomaly score", c="orangered", lw=3)
    (binary_anom * 45 - 150).plot(label="detected binary anomaly", lw=4)
    plt.legend()
    plt.savefig('anomaly_plot.png')
    return "Plot saved as 'anomaly_plot.png'"


if __name__ == '__main__':
    app.run(debug=True)
