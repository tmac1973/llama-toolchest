// Use in the custom set form's Browse dialog (gen_set_browse.html):
// write the chosen file's Hugging Face link into the slot's field, tell
// the form it changed (so the download size is checked again), and
// close the dialog.
function genSetUseFile(button) {
  const dialog = button.closest('dialog');
  // The dialog sits next to the form it serves.
  const form = dialog && dialog.parentElement && dialog.parentElement.querySelector('form');
  const field = form && form.querySelector('input[name="slot_' + button.dataset.slot + '"]');
  if (!field) return;
  field.value = button.dataset.link;
  field.dispatchEvent(new Event('change', { bubbles: true }));
  dialog.close();
}

if (typeof module !== 'undefined') module.exports = { genSetUseFile };
