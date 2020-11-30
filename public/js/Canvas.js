class Canvas {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.points = [];

        this.setWidth(this.getWidth());
        this.setHeight(this.getHeight());
    }

    getWidth() {
        return this.canvas.getBoundingClientRect().width;
    }

    getHeight() {
        return this.canvas.getBoundingClientRect().height;
    }

    getPoints() {
        return this.points;
    }

    setWidth(width) {
        this.canvas.width = width;
    }

    setHeight(height) {
        this.canvas.height = height;
    }

    addPoint(point) {
        this.points.push(point);
        this.drawPoint(point);
    }

    addBackground() {
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.getWidth(), this.getHeight());
    }

    drawPoint(point) {
        this.ctx.beginPath();
        this.ctx.arc(
            point.getX() * this.getWidth(),
            point.getY() * this.getHeight(),
            5,
            0,
            2 * Math.PI
        );
        this.ctx.fillStyle = 'rgb(255, 255, 255)';
        this.ctx.fill();
    }

    drawPoints() {
        for (let i = 0; i < this.points.length; i++) {
            this.drawPoint(this.points[i]);
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.getWidth(), this.getHeight());
    }

    getClickPosition(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        this.addPoint(new Point(x / this.getWidth(), y / this.getHeight()));
    }
}