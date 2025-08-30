
import { LitElement, html } from 'lit-element';

class HelloComponent extends LitElement {
  static get properties() { // JavaScript way to define custom properties
    return { name: { type: String } };
  }

  constructor() {
    super();
    this.name = 'Luis'; // Default value goes here.
  }

  render() { // Defines a template to be "rendered" as part of the component.
    return html`Hello ${this.name}. Welcome to LitElement!`;
  }
}

// Register as a custom element named <hello-component>
customElements.define('hello-component', MyElement);
