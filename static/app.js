(() => {
"use strict";
// JavaScript управляет только формой. Прогноз выполняют сохранённые Python-модели.
const form = document.getElementById("prediction-form");
const fields = [...form.querySelectorAll("input:not([type='hidden']), select")];
function invalidateResult() {
  const result = document.getElementById("prediction-result");
  if (result) {
    result.hidden = true;
    document.getElementById("stale-result").hidden = false;
  }
}
form.addEventListener("input", invalidateResult);
form.addEventListener("change", invalidateResult);
document.getElementById("fill-example").addEventListener("click", () => {
  for (const field of fields) {
    field.value = field.tagName === "SELECT" ? String(Number(field.dataset.example)) : field.dataset.example;
  }
  invalidateResult();
});
document.getElementById("clear-form").addEventListener("click", () => {
  for (const field of fields) field.value = "";
  invalidateResult();
  fields[0].focus();
});

})();
