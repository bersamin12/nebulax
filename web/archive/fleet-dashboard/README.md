# Fleet replay dashboard archive

This folder contains the former `?page=twin` fleet replay dashboard and its dashboard-only
components. It was removed from the active application after the product was reduced to two
user-facing pages: Overview and Predict.

`LegacyApp.jsx` is a snapshot of the application shell before removal and includes the original
`TwinConsole` implementation. The other files retain the replay panels, transport controls, and
state stores that supported it. Git history remains the authoritative source if this feature is
restored.

The 3D train viewport was not archived. It remains under `web/src/components/viewport/` because
the prediction workflow uses it to display selected results.

Old `?page=twin`, `?train=…`, and `?component=…` links now open the prediction workflow.
