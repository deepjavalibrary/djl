import { configure } from '@storybook/react';
import '@patternfly/react-core/dist/styles/base.css';

// automatically import all files ending in *.stories.tsx
const req = require.context('../stories', true, /\.stories\.tsx$/);

function loadStories() {
  req.keys().forEach(req);
}

configure(loadStories, module);
