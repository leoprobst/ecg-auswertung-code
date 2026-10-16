"""Open one blinded reviewer packet, separate from the coordinator GUI."""
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from .mimic.reviewer_packet import ReviewerPacket
from .gui.widgets.reviewer_signal_widget import ReviewerSignalWidget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet",type=Path,required=True)
    parser.add_argument("--responses",type=Path,required=True)
    args = parser.parse_args()
    packet = ReviewerPacket(args.packet)
    app = QApplication([])
    widget = ReviewerSignalWidget(packet,args.responses)
    widget.setWindowTitle("Verblindete EKG-Beurteilung · Forschung")
    widget.resize(1366,900)
    widget.show()
    app.exec()


if __name__ == "__main__":
    main()
