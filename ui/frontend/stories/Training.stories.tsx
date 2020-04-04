import * as React from 'react';
import { storiesOf } from '@storybook/react';
import { withInfo } from '@storybook/addon-info';
import { Training } from '@app/Training/Training';

const stories = storiesOf('Components', module);
stories.addDecorator(withInfo);
stories.add(
  'Training',
  () => <Training />,
  { info: { inline: true } }
);
