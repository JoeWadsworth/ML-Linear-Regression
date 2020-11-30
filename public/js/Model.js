class Model {
    constructor(canvas) {
        this.canvas = canvas;
        this.animation;

        this.lossText = document.getElementById("loss");
        this.epochText = document.getElementById("epoch");

        // TensorFlow dependant variables
        this.a = tf.variable(tf.scalar(Math.random())); // Random number to start
        this.b = tf.variable(tf.scalar(Math.random())); // Random number to start
        this.c = tf.variable(tf.scalar(Math.random())); // Random number to start
        this.learningRate = 0.3;
        this.optimizer = tf.train.adam(this.learningRate);

        // Stats
        this.epoch = 0;
        this.lossGraph;
        this.maxLoss = 0;
        this.currentLoss;
    }

    initialise() {
        this.canvas.addBackground();
        this.createLossGraph();
        this.startAnimation();
    }

    iteration() {
        if (this.canvas.points.length > 0) {
            this.epoch++
            this.canvas.clear();
            this.canvas.addBackground();
            this.canvas.drawPoints();

            tf.tidy(() => {
                this.minimizeLoss();
            });

            //this.drawLine();
            this.drawCurve();
            this.updateStats();
            this.updateLossGraph();

            //console.log(tf.memory().numTensors);
        }
        this.startAnimation();
    }

    startAnimation() {
        this.animation = setTimeout(this.iteration.bind(this), 50);
    }

    stopAnimation() {
        window.clearTimeout(this.animation);
    }

    // ########## TensorFlow Section ##############

    getXValues() {
        let xs = [];
        for (let i = 0; i < this.canvas.points.length; i++) {
            xs.push(this.canvas.points[i].x);
        }
        return xs
    }

    getYValues() {
        let ys = [];
        for (let i = 0; i < this.canvas.points.length; i++) {
            ys.push(this.canvas.points[i].y);
        }
        return ys
    }

    drawLine() {
        if (this.canvas.points.length > 0) {
            const lineX = [0, 1];
            const ys = tf.tidy(() => this.predict(lineX));
            let lineY = ys.dataSync();
            ys.dispose();

            this.canvas.ctx.beginPath();
            this.canvas.ctx.moveTo(lineX[0] * this.canvas.getWidth(), lineY[0] * this.canvas.getHeight());
            this.canvas.ctx.lineTo(lineX[1] * this.canvas.getWidth(), lineY[1] * this.canvas.getHeight());
            this.canvas.ctx.strokeStyle = '#fff';
            this.canvas.ctx.stroke();
        }
    }

    drawCurve() {
        if (this.canvas.points.length > 0) {
            const curveX = [];
            for (let x = 0; x <= 20; x++) {
                curveX.push(x / 20);
            }

            console.log(curveX);

            const ys = tf.tidy(() => this.predict(curveX));
            let curveY = ys.dataSync();
            ys.dispose();

            this.canvas.ctx.beginPath();
            this.canvas.ctx.moveTo(curveX[0] * this.canvas.getWidth(), curveY[0] * this.canvas.getHeight());

            for (let i = 0; i < curveX.length; i++) {
                this.canvas.ctx.lineTo(curveX[i+1] * this.canvas.getWidth(), curveY[i+1] * this.canvas.getHeight());
            }
            this.canvas.ctx.strokeStyle = '#fff';
            this.canvas.ctx.stroke();
        }
    }

    predict(x) {
        const xs = tf.tensor1d(x);
        //const ys = xs.mul(this.a).add(this.b); // y = ax + b
        const ys = xs.square().mul(this.a).add(xs.mul(this.b)).add(this.c); // y = ax^2 + bx + c
        return ys;
    }

    loss = function(pred, labels) {
        return pred.sub(labels).square().mean();
    }

    minimizeLoss() {
        if (this.canvas.points.length > 0) {
            const ys = tf.tensor1d(this.getYValues());
            this.optimizer.minimize(() => this.loss(this.predict(this.getXValues()), ys));
            this.currentLoss = this.loss(this.predict(this.getXValues()), ys).dataSync()[0];
        }
    }

    updateStats() {
        if (this.canvas.points.length > 0) {
            this.lossText.innerHTML = `Loss: ${this.currentLoss}`;
            this.epochText.innerHTML = `Epoch: ${this.epoch}`;

            if (this.currentLoss[0] > this.maxLoss) {
                this.maxLoss = this.currentLoss[0];
            }
        }
    }

    createLossGraph() {
        let ctx = document.getElementById('lossGraph').getContext('2d');

        let chartColors = {
            red: 'rgb(255, 99, 132)',
            orange: 'rgb(255, 159, 64)',
            yellow: 'rgb(255, 205, 86)',
            green: 'rgb(75, 192, 192)',
            blue: 'rgb(54, 162, 235)',
            purple: 'rgb(153, 102, 255)',
            grey: 'rgb(201, 203, 207)'
        };

        let config = {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    backgroundColor: chartColors.red,
                    borderColor: chartColors.red,
                    data: [],
                    fill: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                title: {
                    display: false
                },
                legend: {
                    display: false
                },
                tooltips: {
                    mode: 'index',
                    intersect: false,
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                },
                scales: {
                    xAxes: [{
                        display: false
                    }],
                    yAxes: [{
                        display: false,
                        ticks: {
                            beginAtZero: true,
                            steps: 0.1,
                            stepValue: 0.5
                        }
                    }]
                }
            }
        };

        this.lossGraph = new Chart(ctx, config);
    }

    updateLossGraph() {
        this.lossGraph.data.labels[this.epoch-1] = this.epoch;
        this.lossGraph.data.datasets[0].data[this.epoch-1] = this.currentLoss;

        if (this.currentLoss[0] === this.maxLoss) {
            this.lossGraph.options.scales.yAxes[0].ticks.max = this.maxLoss;
        }

        this.lossGraph.update();
    }
}
