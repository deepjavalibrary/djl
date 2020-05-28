
var input = "import ai.djl.ndarray.NDArray;\nimport ai.djl.ndarray.NDManager;\nimport ai.djl.ndarray.types.Shape;\nimport ai.djl.ndarray.index.NDIndex;\n\nNDManager manager = NDManager.newBaseManager();\nNDArray array = manager.ones(new Shape(1,3,2));"

var editor = CodeMirror.fromTextArea(document.getElementById("editor"), {
  mode: "text/x-java",
  theme: "dracula",
  lineNumbers: true,
  matchBrackets: true
});

var result = CodeMirror.fromTextArea(document.getElementById("result"), {
  theme: "dracula"
});

editor.setValue(input);

function submitCode() {
    var value = editor.getValue();
    var select = document.getElementById("engine");
    var engine = select.options[select.selectedIndex].value;
    const Url = "https://olzo20ie3f.execute-api.us-east-1.amazonaws.com/DJL-Block-Runner";
    result.setValue("Running in progress...");
    fetch(Url, {
        method: "POST",
        headers: {
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: JSON.stringify({ "engine" : engine, "commands" : value})
        }).then(response => response.json())
        .then(data => {
          var resultString = data["result"];
          result.setValue(resultString);
        })
        .catch((error) => {
          console.error("Error:", error)
          result.setValue(error.toString());
        });
}

