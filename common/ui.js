'use strict';

/* eslint max-len: ["error", {"code": 350}] */

export async function showProgressComponent(pm, pb, pi) {
  let p = '';
  let modelicon = ``;
  if (pm === 'done') {
    modelicon = `<svg class='prog_list_icon' viewbox='0 0 24 24'>
                    <path class='st0' d='M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8zm0 1.5c-5.2 0-9.5-4.3-9.5-9.5S6.8 2.5 12 2.5s9.5 4.3 9.5 9.5-4.3 9.5-9.5 9.5z'></path>
                    <path class='st0' d='M11.1 12.9l-1.2-1.1c-.4-.3-.9-.3-1.3 0-.3.3-.4.8-.1 1.1l.1.1 1.8 1.6c.1.1.4.3.7.3.2 0 .5-.1.7-.3l3.6-4.1c.3-.3.4-.8.1-1.1l-.1-.1c-.4-.3-1-.3-1.3 0l-3 3.6z'></path>
                  </svg>`;
  } else if (pm === 'current') {
    modelicon = `<svg class='prog_list_icon prog_list_icon-${pb}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  } else {
    modelicon = `<svg class='prog_list_icon prog_list_icon-${pi}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12 16.1c1.8 0 3.3-1.4 3.3-3.2 0-1.8-1.5-3.2-3.3-3.2s-3.3 1.4-3.3 3.2c0 1.7 1.5 3.2 3.3 3.2zm0 1.7c-2.8 0-5-2.2-5-4.9S9.2 8 12 8s5 2.2 5 4.9-2.2 4.9-5 4.9z'></path>
                </svg>`;
  }

  let updateicon = ``;
  if (pb === 'done') {
    updateicon = `<svg class='prog_list_icon' viewbox='0 0 24 24'>
                    <path class='st0' d='M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8zm0 1.5c-5.2 0-9.5-4.3-9.5-9.5S6.8 2.5 12 2.5s9.5 4.3 9.5 9.5-4.3 9.5-9.5 9.5z'></path>
                    <path class='st0' d='M11.1 12.9l-1.2-1.1c-.4-.3-.9-.3-1.3 0-.3.3-.4.8-.1 1.1l.1.1 1.8 1.6c.1.1.4.3.7.3.2 0 .5-.1.7-.3l3.6-4.1c.3-.3.4-.8.1-1.1l-.1-.1c-.4-.3-1-.3-1.3 0l-3 3.6z'></path>
                  </svg>`;
  } else if (pb === 'current') {
    updateicon = `<svg class='prog_list_icon prog_list_icon-${pb}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  } else {
    updateicon = `<svg class='prog_list_icon prog_list_icon-${pi}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12 16.1c1.8 0 3.3-1.4 3.3-3.2 0-1.8-1.5-3.2-3.3-3.2s-3.3 1.4-3.3 3.2c0 1.7 1.5 3.2 3.3 3.2zm0 1.7c-2.8 0-5-2.2-5-4.9S9.2 8 12 8s5 2.2 5 4.9-2.2 4.9-5 4.9z'></path>
                </svg>`;
  }

  let inferenceicon = ``;
  if (pi === 'done') {
    inferenceicon = `<svg class='prog_list_icon' viewbox='0 0 24 24'>
                    <path class='st0' d='M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8zm0 1.5c-5.2 0-9.5-4.3-9.5-9.5S6.8 2.5 12 2.5s9.5 4.3 9.5 9.5-4.3 9.5-9.5 9.5z'></path>
                    <path class='st0' d='M11.1 12.9l-1.2-1.1c-.4-.3-.9-.3-1.3 0-.3.3-.4.8-.1 1.1l.1.1 1.8 1.6c.1.1.4.3.7.3.2 0 .5-.1.7-.3l3.6-4.1c.3-.3.4-.8.1-1.1l-.1-.1c-.4-.3-1-.3-1.3 0l-3 3.6z'></path>
                  </svg>`;
  } else if (pi === 'current') {
    inferenceicon = `<svg class='prog_list_icon prog_list_icon-${pb}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  } else {
    inferenceicon = `<svg class='prog_list_icon prog_list_icon-${pi}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12 16.1c1.8 0 3.3-1.4 3.3-3.2 0-1.8-1.5-3.2-3.3-3.2s-3.3 1.4-3.3 3.2c0 1.7 1.5 3.2 3.3 3.2zm0 1.7c-2.8 0-5-2.2-5-4.9S9.2 8 12 8s5 2.2 5 4.9-2.2 4.9-5 4.9z'></path>
                </svg>`;
  }

  p = `
      <nav class='prog'>
        <ul class='prog_list'>
          <li class='prog prog-${pm}'>
            ${modelicon}<span class='prog_list_title'>Model loading</span>
          </li>
          <li class='prog prog-${pb}'>
            ${updateicon}<span class='prog_list_title'>Model building</span>
          </li>
          <li class='prog prog-${pi}'>
            ${inferenceicon}<span class='prog_list_title'>Model inferencing</span>
          </li>
        </ul>
      </nav>
    `;

  $('#progressmodel').show();
  $('#progressstep').html(p);
  $('.shoulddisplay').hide();
  $('.icdisplay').hide();
  await new Promise((res) => setTimeout(res, 100));
}

export function readyShowResultComponents() {
  $('#progressmodel').hide();
  $('.icdisplay').show();
  $('.shoulddisplay').show();
}

// Handle buttons click
// Use to disable buttons click during model running and resume them once
// model running done
export function handleClick(cssSelectors, disabled = true) {
  /* eslint no-unused-vars: ["error", { "varsIgnorePattern": "selector" }] */
  for (const selector of cssSelectors) {
    if (disabled) {
      $(selector).addClass('clickDisabled');
      if (selector.startsWith('.btn')) $(selector).addClass('styleDisabled');
    } else {
      $(selector).removeClass('clickDisabled');
      if (selector.startsWith('.btn')) $(selector).removeClass('styleDisabled');
    }
  }
}

/**
 * Show flexible alert messages
 * @param {String} msg, alert message.
 * @param {String} type, one of ["info", "warning"], type of message,
 * default is 'warning'
 */
export function addAlert(msg, type = 'warning') {
  let alertClass = 'alert-warning';
  if (type === 'info') {
    alertClass = 'alert-info';
    if ($('.alert-info').length) {
      $('.alert-info > span').html(msg);
      return;
    }
  }

  $('<div>', {
    'class': `alert ${alertClass} alert-dismissible fade show mt-3`,
    'role': 'alert',
    'html': `<span>${msg}</span>`,
  }).append($('<button>', {
    'type': 'button',
    'class': 'close',
    'data-dismiss': 'alert',
    'aria-label': 'close',
    'html': '<span aria-hidden="true">&times;</span>',
  })).insertBefore($('#container').children()[0]);
}
