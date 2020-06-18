import ai.djl.ndarray.*;
import ai.djl.training.dataset.*;
import ai.djl.basicdataset.FashionMnist;
import ai.djl.translate.TranslateException;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Component;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.BoxLayout;

class FashionMnistUtils {

    public static RandomAccessDataset getDataset(Dataset.Usage usage,
                                                 int batchSize,
                                                 boolean randomShuffle) throws IOException {
        FashionMnist fashionMnist = FashionMnist.builder().optUsage(usage)
                .setSampling(batchSize, randomShuffle)
                .build();
        fashionMnist.prepare();
        return fashionMnist;
    }

    public static String[] getFashionMnistLabels(int[] labelIndices) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        String[] convertedLabels = new String[labelIndices.length];
        for (int i = 0; i < labelIndices.length; i++) {
            convertedLabels[i] = textLabels[labelIndices[i]];
        }
        return convertedLabels;
    }

    public static String getFashionMnistLabel(int labelIndice) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        return textLabels[labelIndice];
    }

    public static class ImagePanel extends JPanel {
        int SCALE;
        BufferedImage img;

        public ImagePanel() {
            this.SCALE = 1;
        }

        public ImagePanel(int scale, BufferedImage img) {
            this.SCALE = scale;
            this.img = img;
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2d = (Graphics2D) g;
            g2d.scale(SCALE, SCALE);
            g2d.drawImage(this.img, 0, 0, this);
        }
    }

    public static class Container extends JPanel {
        public Container(String label) {
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            JLabel l = new JLabel(label, JLabel.CENTER);
            l.setAlignmentX(Component.CENTER_ALIGNMENT);
            add(l);
        }

        public Container(String trueLabel, String predLabel) {
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            JLabel l = new JLabel(trueLabel, JLabel.CENTER);
            l.setAlignmentX(Component.CENTER_ALIGNMENT);
            add(l);
            JLabel l2 = new JLabel(predLabel, JLabel.CENTER);
            l2.setAlignmentX(Component.CENTER_ALIGNMENT);
            add(l2);
        }
    }

    public static void showImages(RandomAccessDataset dataset,
                                  int number, int WIDTH, int HEIGHT, int SCALE,
                                  NDManager manager)
            throws IOException, TranslateException {
        // Plot a list of images
        JFrame frame = new JFrame("Fashion Mnist");
        for (int record = 0; record < number; record++) {
            NDArray X = dataset.get(manager, record).getData().get(0).squeeze(-1);
            int y = (int) dataset.get(manager, record).getLabels().get(0).getFloat();
            BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g = (Graphics2D) img.getGraphics();
            for (int i = 0; i < WIDTH; i++) {
                for (int j = 0; j < HEIGHT; j++) {
                    float c = X.getFloat(j, i) / 255;  // scale down to between 0 and 1
                    g.setColor(new Color(c, c, c)); // set as a gray color
                    g.fillRect(i, j, 1, 1);
                }
            }
            JPanel panel = new ImagePanel(SCALE, img);
            panel.setPreferredSize(new Dimension(WIDTH * SCALE, HEIGHT * SCALE));
            JPanel container = new Container(getFashionMnistLabel(y));
            container.add(panel);
            frame.getContentPane().add(container);
        }
        frame.getContentPane().setLayout(new FlowLayout());
        frame.pack();
        frame.setVisible(true);
    }

    public static void showImages(RandomAccessDataset dataset, int[] predLabels,
                                  int WIDTH, int HEIGHT, int SCALE,
                                  NDManager manager)
            throws IOException, TranslateException {
        // Plot a list of images
        JFrame frame = new JFrame("Fashion Mnist");
        for (int record = 0; record < predLabels.length; record++) {
            NDArray X = dataset.get(manager, record).getData().get(0).squeeze(-1);
            int y = (int) dataset.get(manager, record).getLabels().get(0).getFloat();
            BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g = (Graphics2D) img.getGraphics();
            for (int i = 0; i < WIDTH; i++) {
                for (int j = 0; j < HEIGHT; j++) {
                    float c = X.getFloat(j, i) / 255;  // scale down to between 0 and 1
                    g.setColor(new Color(c, c, c)); // set as a gray color
                    g.fillRect(i, j, 1, 1);
                }
            }
            JPanel panel = new ImagePanel(SCALE, img);
            panel.setPreferredSize(new Dimension(WIDTH * SCALE, HEIGHT * SCALE));
            JPanel container = new Container(getFashionMnistLabel(y), getFashionMnistLabel(predLabels[i]);
            container.add(panel);
            frame.getContentPane().add(container);
        }
        frame.getContentPane().setLayout(new FlowLayout());
        frame.pack();
        frame.setVisible(true);
    }
}

