import { LitElement, html, property, customElement } from 'lit-element';

@customElement('hello-component') //Decorator that register as a custom element named <hello-component>
export class HelloComponent extends LitElement {
  @property({type: String}) name = 'Luis'; // You can assign the default value here.

  render() { // Defines a template to be "rendered" as part of the component.
    return html`Hello, ${this.name}. Welcome to LitElement!</p>`;
  }
}
