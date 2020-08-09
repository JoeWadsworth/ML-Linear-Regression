function Point(x, y) {
  this.x = x;
  this.y = y;
}

function Model(width, height) {
  this.lossText = document.getElementById("loss");
  this.canvas = document.getElementById("canvas");
  this.ctx = this.canvas.getContext("2d");
  this.canvas.width = width;
  this.canvas.height = height;
  // this.width = width
  // this.height = height;
  this.points = [];
  this.animation;
  this.iter = 0;

  // TensorFlow dependant variables
  this.a = tf.variable(tf.scalar(Math.random())); // Random number to start
  this.b = tf.variable(tf.scalar(Math.random())); // Random number to start
  this.c = tf.variable(tf.scalar(Math.random())); // Random number to start
  this.learningRate = 0.3;
  this.optimizer = tf.train.adam(this.learningRate);
}

Model.prototype.initialise = function () {
  this.drawCanvas();
  //this.createPoints();
  this.drawPoints();

  tf.tidy(() => {
    this.minimizeLoss();
  });
  
  this.startAnimation();
}

Model.prototype.drawCanvas = function () {
  this.ctx.fillStyle = "black";
  this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
}

Model.prototype.clearCanvas = function () {
  this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
}

Model.prototype.createPoints = function () {
  for (i = 0; i < 4; i++) {
    this.addPoint(new Point(Math.random(), Math.random()));
  }
}

Model.prototype.drawPoints = function () {
  for (i = 0; i < this.points.length; i++) {
    this.drawPoint(this.points[i].x * this.canvas.width, this.points[i].y * this.canvas.height);
  }
}

Model.prototype.drawPoint = function (x, y) {
  this.ctx.beginPath();
  this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
  this.ctx.fillStyle = 'rgb(255, 255, 255)';
  this.ctx.fill();
}

Model.prototype.addPoint = function (point) {
  this.points.push(point);
}

Model.prototype.iteration = function () {
  this.iter++
  this.clearCanvas();
  this.drawCanvas();
  this.drawPoints();

  tf.tidy(() => {
    this.minimizeLoss();
  });

  //this.drawLine();
  this.drawCurve();

  //console.log(tf.memory().numTensors);

  this.startAnimation();
}

Model.prototype.startAnimation = function() {
  this.animation = setTimeout(Model.prototype.iteration.bind(this), 50);
}

Model.prototype.stopAnimation = function() {
  window.clearTimeout(this.animation);
}

// ########## TensorFlow Section ##############

Model.prototype.getXValues = function () {
  let xs = [];
  for (i = 0; i < this.points.length; i++) {
    xs.push(this.points[i].x);
  }
  return xs
}

Model.prototype.getYValues = function () {
  let ys = [];
  for (i = 0; i < this.points.length; i++) {
    ys.push(this.points[i].y);
  }
  return ys
}

Model.prototype.drawLine = function () {
  if (this.points.length > 0) {
    const lineX = [0, 1];
    const ys = tf.tidy(() => this.predict(lineX));
    let lineY = ys.dataSync();
    ys.dispose();
  
    this.ctx.beginPath();
    this.ctx.moveTo(lineX[0] * this.canvas.width, lineY[0] * this.canvas.height);
    this.ctx.lineTo(lineX[1] * this.canvas.width, lineY[1] * this.canvas.height);
    this.ctx.strokeStyle = '#fff';
    this.ctx.stroke();
  }
}

Model.prototype.drawCurve = function () {
  if (this.points.length > 0) {
    const curveX = [];
    for (x = 0; x <= 1; x += 0.05) {
      curveX.push(x);
    }

    const ys = tf.tidy(() => this.predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();

    this.ctx.beginPath();
    this.ctx.moveTo(curveX[0] * this.canvas.width, curveY[0] * this.canvas.height);

    for (i = 0; i < curveX.length; i++) {
      this.ctx.lineTo(curveX[i+1] * this.canvas.width, curveY[i+1] * this.canvas.height);
    }
    this.ctx.strokeStyle = '#fff';
    this.ctx.stroke();
  }
}

Model.prototype.predict = function (x) {
  const xs = tf.tensor1d(x);
  //const ys = xs.mul(this.a).add(this.b); // y = ax + b
  const ys = xs.square().mul(this.a).add(xs.mul(this.b)).add(this.c); // y = ax^2 + bx + c
  return ys;
}

Model.prototype.loss = function(pred, labels) {
  return pred.sub(labels).square().mean();
}

Model.prototype.minimizeLoss = function () {
  if (this.points.length > 0) {
    const ys = tf.tensor1d(this.getYValues());
    this.optimizer.minimize(() => this.loss(this.predict(this.getXValues()), ys));

    const loss = this.loss(this.predict(this.getXValues()), ys).dataSync();
    this.lossText.innerHTML = `Loss: ${loss}`;
  }
}

Model.prototype.getClickPosition = function (event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    this.addPoint(new Point(x / this.canvas.width, y / this.canvas.height));
    this.drawPoint(x,y);
}

$(document).ready(function () {
  let canvas = document.getElementById("canvas");
  console.log(canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);
  let model = new Model(canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);

  

  model.canvas.onclick = function (e) {
    model.getClickPosition(e);
  };

  model.initialise();
});