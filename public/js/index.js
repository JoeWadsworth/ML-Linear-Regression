$(document).ready(function () {
  let canvas = new Canvas(document.getElementById('canvas'));
  let model = new Model(canvas);

  document.getElementById('canvas').addEventListener('click', event => {
    canvas.getClickPosition(event);
  });

  model.initialise();
});