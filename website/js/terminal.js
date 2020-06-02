// https://medium.com/codingtown/xterm-js-terminal-2b19ccd2a52
// https://medium.com/swlh/local-echo-xterm-js-5210f062377e
var term = new Terminal();
const consoleId = (Math.random() + 1).toString(36).substring(7);

term.open(document.getElementById('terminal'));
term.setOption('cursorBlink', true);

const prefix = "djl.ai@jconsole> ";
var input = "";
var cursor = 0;

function init() {
  term.reset();
  term.write("Welcome to the simulated jconsole for DJL.\r\nThis console is equipped with:\r\nNDManager manager\r\n");
  term.write(prefix);
}

init();

term.on("data", (data) => {
  const code = data.charCodeAt(0);
  if (code == 13) { // Enter
    analyseResponse(input);
    cursor = 0;
  } else if(code == 127) { // Backspace
    if (input.length > 0 && cursor > 0) {
      input = input.substr(0, cursor - 1) + input.substr(cursor);
      cursor -= 1;
      rewriteInput(term, input, cursor);
    }
  } else if (code < 32) { // Control
        switch (data.substr(1)) {
          case '[C': // Right arrow
            if (cursor < input.length) {
              cursor += 1;
              term.write(data);
            }
            break;
          case '[D': // Left arrow
            if (cursor > 0) {
              cursor -= 1;
              term.write(data);
            }
            break;
        }
  } else { // Visible
    input = input.substr(0, cursor) +
                data +
                input.substr(cursor);
    cursor += 1;
    if (cursor != input.length) {
        rewriteInput(term, input, cursor);
    } else {
        term.write(data);
    }
  }
});

function rewriteInput(term, input, cursor) {
    // refer: http://www.climagic.org/mirrors/VT100_Escape_Codes.html
    term.write('\x1b[2K'); // clean entire line
    term.write('\r' + prefix); // rewrite prefix
    term.write(input); // write content
    if (input.length - cursor > 0) {
        term.write('\x1b[' + (input.length - cursor) + 'D'); // move cursor back
    }
}

function analyseResponse(data) {
  if (data == "clear") {
    init();
    input = "";
  }
  else {
    const Url = "https://demo.djl.ai/addCommand";
    fetch(Url, {
    method: "POST",
    headers: {
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: JSON.stringify({ "console_id" : consoleId, "command" : data})
    }).then(response => response.json())
    .then(data => {
      if (data["result"].length > 0) {
        var resultString = data["result"].split("\n").join("\r\n")
        term.write("\r\n" + resultString + "\r\n");
        term.write(prefix);
      } else {
        term.write("\r\n" + prefix);
      }
      input = "";
    })
    .catch((error) => {
      console.error("Error:", error)
      term.write("\r\n" + error + "\r\n");
      term.write(prefix);
      input = "";
    });
  }
}
