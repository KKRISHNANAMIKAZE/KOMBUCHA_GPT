import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class ChatScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("K-GPT")),
      body: WebView(
        initialUrl: "https://huggingface.co/spaces/krishna4/kgpt/tree/main",
        javascriptMode: JavascriptMode.unrestricted,
      ),
    );
  }
}S