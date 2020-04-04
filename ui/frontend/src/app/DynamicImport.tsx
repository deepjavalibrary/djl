import * as React from 'react';
import { accessibleRouteChangeHandler } from '@app/utils/utils';

interface IDynamicImport {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  load: () => Promise<any>;
  children: any;
  /* eslint-enable @typescript-eslint/no-explicit-any */
  focusContentAfterMount: boolean;
}

class DynamicImport extends React.Component<IDynamicImport> {
  public state = {
    component: null
  };
  private routeFocusTimer: number;
  constructor(props: IDynamicImport) {
    super(props);
    this.routeFocusTimer = 0;
  }
  public componentWillUnmount() {
    window.clearTimeout(this.routeFocusTimer);
  }
  public componentDidMount() {
    this.props
      .load()
      .then(component => {
        if (component) {
          this.setState({
            component: component.default ? component.default : component
          });
        }
      })
      .then(() => {
        if (this.props.focusContentAfterMount) {
          this.routeFocusTimer = accessibleRouteChangeHandler();
        }
      });
  }
  public render() {
    return this.props.children(this.state.component);
  }
}

export { DynamicImport };
