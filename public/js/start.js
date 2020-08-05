function Point(x, y) {
  this.x = x;
  this.y = y;
}

function Canvas(width, height) {
  this.canvas = document.getElementById("canvas");
  this.ctx = this.canvas.getContext("2d");
  this.canvas.width = width;
  this.canvas.height = height;
  this.width = width
  this.height = height;
  this.points = [];
  this.animation;
  this.iter = 0;

  // TensorFlow dependant variables
  this.m = tf.variable(tf.scalar(Math.random())); // Random number to start
  this.c = tf.variable(tf.scalar(Math.random())); // Random number to start
  this.learningRate = 0.2;
  this.optimizer = tf.train.sgd(this.learningRate);
}

Canvas.prototype.initialise = function () {
  this.drawCanvas();
  this.createPoints();
  this.drawPoints();

  tf.tidy(() => {
    this.minimizeLoss();
  });
  
  this.startAnimation();
}

Canvas.prototype.drawCanvas = function () {
  this.ctx.fillStyle = "black";
  this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
}

Canvas.prototype.clearCanvas = function () {
  this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
}

Canvas.prototype.createPoints = function () {
  for (i = 0; i < 4; i++) {
    this.addPoint(new Point(Math.random(), Math.random()));
  }
}

Canvas.prototype.drawPoints = function () {
  for (i = 0; i < this.points.length; i++) {
    this.drawPoint(this.points[i].x * this.canvas.width, this.points[i].y * this.canvas.height);
  }
}

Canvas.prototype.drawPoint = function (x, y) {
  this.ctx.beginPath();
  this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
  this.ctx.fillStyle = 'rgb(255, 255, 255)';
  this.ctx.fill();
}

Canvas.prototype.addPoint = function (point) {
  this.points.push(point);
}

Canvas.prototype.iteration = function () {
  this.iter++
  this.clearCanvas();
  this.drawCanvas();
  this.drawPoints();

  tf.tidy(() => {
    this.minimizeLoss();
  });

  this.drawLine();

  //console.log(tf.memory().numTensors);

  this.startAnimation();
}

Canvas.prototype.startAnimation = function() {
  this.animation = setTimeout(Canvas.prototype.iteration.bind(this), 50);
}

Canvas.prototype.stopAnimation = function() {
  window.clearTimeout(this.animation);
}


// ########## TensorFlow Section ##############

Canvas.prototype.getXValues = function () {
  let xs = [];
  for (i = 0; i < this.points.length; i++) {
    xs.push(this.points[i].x);
  }
  return xs
}

Canvas.prototype.getYValues = function () {
  let ys = [];
  for (i = 0; i < this.points.length; i++) {
    ys.push(this.points[i].y);
  }
  return ys
}

Canvas.prototype.drawLine = function () {
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

Canvas.prototype.predict = function (x) {
  const xs = tf.tensor1d(x);
  const ys = xs.mul(this.m).add(this.c); // y = mx + c
  return ys;
}

Canvas.prototype.loss = function(pred, labels) {
  return pred.sub(labels).square().mean();
}

Canvas.prototype.minimizeLoss = function () {
  if (this.points.length > 0) {
    const ys = tf.tensor1d(this.getYValues());
    this.optimizer.minimize(() => this.loss(this.predict(this.getXValues()), ys));
  }
}

Canvas.prototype.getClickPosition = function (event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    this.addPoint(new Point(x / this.canvas.width, y / this.canvas.height));
    this.drawPoint(x,y);
}

$(document).ready(function () {
  let canvas = new Canvas(window.innerWidth, window.innerHeight - 56, 15);

  canvas.canvas.onclick = function (e) {
    canvas.getClickPosition(e);
  };

  canvas.initialise();
});