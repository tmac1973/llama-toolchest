# Scalar API Reference

`standalone.js` is the browser build of
[@scalar/api-reference](https://www.npmjs.com/package/@scalar/api-reference)
version 1.73.1 (`dist/browser/standalone.js` in the npm package), MIT
licensed. It draws the API reference page at `/docs/api` from
`/openapi.json`. It is kept here, like htmx and Plotly, so the page works
without internet access.

To update it, run `npm pack @scalar/api-reference@<version>`, copy
`package/dist/browser/standalone.js` over this one, change the version above,
and check `/docs/api` still loads with no requests to scalar.com (open the
browser's network tab).
