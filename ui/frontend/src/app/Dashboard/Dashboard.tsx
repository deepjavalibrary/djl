import * as React from 'react';
import { Page, PageSection, PageSectionVariants, TextContent, Text, Grid, GridItem, Card, CardHeader, CardBody, Progress, ProgressSize, ProgressMeasureLocation, ProgressVariant } from '@patternfly/react-core';
import { Chart, ChartAxis, ChartGroup, ChartLine, ChartVoronoiContainer } from '@patternfly/react-charts';
import * as EventBus from 'vertx3-eventbus-client';

interface IModelInfo {
  name: string;
  block: string;
}
interface ITrainerInfo {
  modelInfo: IModelInfo;
  devices: [];
  epoch: number;
  trainingProgress: number;
  validatingProgress: number;
  speed: number;
  metricsSize: number;
  state: string;
  metricNames: string[];
  metrics: {[key: string]: IMetricInfo[]};
}
interface IMetricInfo {
  name: string;
  x: number;
  y: number;
}

const Dashboard: React.FunctionComponent = () => {

  const [trainerInfo, setTrainerInfo] = React.useState<ITrainerInfo>();

  const eventBus = new EventBus('/api/eventbus');
  React.useEffect(() => {
    eventBus.enableReconnect(true);
    eventBus.onopen = function () {
      eventBus.registerHandler('trainer', function (error, message) {
        const t: ITrainerInfo = JSON.parse(message.body) as ITrainerInfo;
        console.log(t);
        setTrainerInfo(t);
      });
      eventBus.send('trainer-request', '');
    }

    return () => {
      eventBus.close();
    }
  }, []);

  return (
    <Page>
      <PageSection variant={PageSectionVariants.light}>
        <TextContent>
          <Text component="h1">Dashboard</Text>
          <Text component="p">Training overview</Text>
        </TextContent>
      </PageSection>
      <PageSection>
        <Grid gutter="md">
          <GridItem span={8}>
            <Card>
              <CardHeader>Training performance</CardHeader>
              <CardBody>
                <Chart
                  containerComponent={<ChartVoronoiContainer responsive={true} labels={({ datum }) => `${datum.name}: ${datum.y}`} constrainToVisibleArea />}
                  legendData={trainerInfo?.metricNames.map(k => ({ name: k }))}
                  legendOrientation="horizontal"
                  legendPosition="bottom"
                  height={600}
                  width={1200}
                  domain={{ x: [0, trainerInfo?.metricsSize != undefined ? trainerInfo.metricsSize : 100], y: [0, 1] }}
                  padding={{ bottom: 70, left: 50, right: 30, top: 30 }}
                >
                  <ChartAxis tickValues={[0, trainerInfo?.metricsSize != undefined ? trainerInfo.metricsSize : 100]} />
                  <ChartAxis key="1" dependentAxis showGrid tickValues={[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]} />
                  <ChartGroup>
                    {Object.assign([], trainerInfo?.metricNames).map((k, index) => 
                      <ChartLine key= {index}  data={trainerInfo?.metrics[k]} interpolation="monotoneX"/>
                    )}
                  </ChartGroup>
                </Chart>
              </CardBody>
            </Card>
          </GridItem>
          <GridItem span={4}>
            <Card>
              <CardHeader>Training information</CardHeader>
              <CardBody>Training on: {trainerInfo?.devices}</CardBody>
              <CardBody>Epoch: {trainerInfo?.epoch}</CardBody>
              <CardBody>
                <Progress value={trainerInfo?.trainingProgress} title="Training:" size={ProgressSize.lg} />
              </CardBody>
              <CardBody>
                <Progress value={trainerInfo?.validatingProgress} title="Validating:" size={ProgressSize.lg} />
              </CardBody>
              <CardBody>Speed: {trainerInfo?.speed} images/sec</CardBody>
            </Card>
          </GridItem>
        </Grid>
      </PageSection>
    </Page>
  );
};

export { Dashboard };
