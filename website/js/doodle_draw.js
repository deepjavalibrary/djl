// add touch event: http://bencentra.com/code/2014/12/05/html5-canvas-touch-events.html
// set canvas id to variable
var canvas = document.getElementById("draw");
canvas.height = canvas.width * 0.75;

// get canvas 2D context and set it to the correct size
var ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
startUp();
// add event listeners to specify when functions should be triggered
canvas.addEventListener("mousemove", function (e) {
  draw(e, true);
});
canvas.addEventListener("mousedown", setPosition);
canvas.addEventListener("mouseenter", setPosition);
canvas.addEventListener("mouseup", runInference);

// Set up touch events for mobile, etc
canvas.addEventListener("touchstart", function (e) {
  e.preventDefault();
  var touch = e.touches[0];
  setPosition(touch);
});
canvas.addEventListener("touchend", function (e) {
  e.preventDefault();
  runInference();
});
canvas.addEventListener("touchmove", function (e) {
  e.preventDefault();
  var touch = e.touches[0];
  draw(touch, false);
});

// last known position
var pos = { x: 0, y: 0 };
// new position from mouse events
function setPosition(e) {
  var rect = canvas.getBoundingClientRect();
  pos.x = (e.clientX - rect.left) / (rect.right - rect.left) * canvas.width;
  pos.y = (e.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height;
}

// bounding box
var box = {x1: 10000, y1: 10000, x2: 0, y2: 0}
// update box
function updateBox(x, y) {
  box.x1 = Math.min(box.x1, x);
  box.x2 = Math.max(box.x2, x);
  box.y1 = Math.min(box.y1, y);
  box.y2 = Math.max(box.y2, y);
}

function draw(e, isMouse) {
  if (isMouse && e.buttons !== 1) return; // if mouse is pressed.....

  ctx.beginPath(); // begin the drawing path

  ctx.lineWidth = 5; // width of line
  ctx.lineCap = "round"; // rounded end cap
  ctx.strokeStyle = "white"; // hex color of line

  ctx.moveTo(pos.x, pos.y); // from position
  updateBox(pos.x, pos.y);
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // to position
  updateBox(pos.x, pos.y);

  ctx.stroke(); // draw it!
}

function getCroppedCanvas() {
  var destCanvas = document.createElement("canvas");
  var boxWidth = Math.max(box.y2 - box.y1, box.x2 - box.x1);
  var boxHeight = boxWidth;
  destCanvas.width = 64;
  destCanvas.height = 64;
  destCanvas.getContext("2d").drawImage(
        canvas,
        box.x1,box.y1,boxWidth,boxHeight,  // source rect with content to crop
        0,0,64,64);
  return destCanvas;
}

function clearMe() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// only used when startup
function startUp() {
  const Url = "https://d2f27q6792.execute-api.us-east-1.amazonaws.com/DJL-Doodle-prod";
  fetch(Url, {
    method: "POST",
    headers: {
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: JSON.stringify({ "imageData" : "image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAABH0lEQVR4Xu2UsQ3CQBAE15CT0g9F0Ao90AlV0A4iIzcCWeLRB37ZG3p+HZ8lZnbwoM6foXN+tQQcJT0qOVhRc2CfmSqekiYpuKcWcJD0mlm9CEFWUKBGSftG8ngBb0m7H3xrZayECXgN3JqbTX4fioCl//dN0lnSVdJlk6SNH70EXr82VXCXdOpVAIn7z+IUEAFEAymAuKrDlAIcW8TbFEBc1WFKAY4t4m0KIK7qMKUAxxbxNgUQV3WYUoBji3ibAoirOkwpwLFFvE0BxFUdphTg2CLepgDiqg5TCnBsEW9TAHFVhykFOLaItymAuKrDlAIcW8TbFEBc1WFKAY4t4m0KIK7qMKUAxxbxNgUQV3WYUoBji3ibAoirOkzdF/AFdNwQQf3s4csAAAAASUVORK5CYII="})
  })
}

// https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
function runInference(e) {
  const Url = "https://d2f27q6792.execute-api.us-east-1.amazonaws.com/DJL-Doodle-prod";
  var destCanvas = getCroppedCanvas();
  fetch(Url, {
    method: "POST",
    headers: {
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: JSON.stringify({ "imageData" : destCanvas.toDataURL("image/png")})
  })
  .then(response => response.json())
  .then(data => {
    var responseString = "";
    for (var idx = 0; idx < data.length; idx++) {
      responseString += data[idx]["className"] + ",";
    }
    M.toast({html: responseString}, 2000);
  })
  .catch((error) => {
    console.error("Error:", error)
  });
}
