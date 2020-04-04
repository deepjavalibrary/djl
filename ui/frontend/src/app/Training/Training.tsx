import * as React from 'react';
import { CubesIcon } from '@patternfly/react-icons';
import {
  PageSection,
  Title,
  Button,
  EmptyState,
  EmptyStateVariant,
  EmptyStateIcon,
  EmptyStateBody,
  EmptyStateSecondaryActions
} from '@patternfly/react-core';


const Training: React.FunctionComponent = () => {

  return (
    <EmptyState variant={EmptyStateVariant.full}>
      <EmptyStateIcon icon={CubesIcon} />
      <Title headingLevel="h5" size="lg">
        Not yet implemplemented
    </Title>
      <EmptyStateBody>
        This page is not yet implemented.
    </EmptyStateBody>
    </EmptyState>
  );
};

export { Training };
