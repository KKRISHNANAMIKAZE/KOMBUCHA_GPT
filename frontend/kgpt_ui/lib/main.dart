import 'dart:convert';
import 'dart:typed_data';
import 'package:url_launcher/url_launcher.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'screens/chat_screen.dart';
import 'dart:html' as html;


void main() => runApp(const KGPTApp());

class KGPTApp extends StatelessWidget {
  const KGPTApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ChatScreen(),
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {

  final TextEditingController controller = TextEditingController();
  final ScrollController scrollController = ScrollController();

  List<Map<String, dynamic>> messages = [];

  bool isLoading = false;

  Uint8List? selectedFileBytes;
  String? selectedFileName;

  final String baseUrl = "https://huggingface.co/spaces/krishna4/kgpt/tree/main";
  final String sessionId = "user_session_001";

  // ================= CHAT =================

  Future<void> sendMessage([String? overrideText]) async {

    String userText = overrideText ?? controller.text;

    if (userText.isEmpty) return;

    setState(() {
      messages.add({"role": "user", "text": userText});
      controller.clear();
      isLoading = true;
    });

    try {

      final response = await http.post(
        Uri.parse("$baseUrl/chat"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "message": userText,
          "session_id": sessionId
        }),
      );

      final data = jsonDecode(response.body);

      setState(() {
        messages.add({
          "role": "assistant",
          "text": data["response"],
          "sources": data["sources"] ?? [],
          "suggestions": data["suggestions"] ?? []
        });
      });

    } catch (e) {

      setState(() {
        messages.add({
          "role": "assistant",
          "text": "⚠️ Backend error. Check server logs."
        });
      });

    }

    setState(() => isLoading = false);

    scrollToBottom();
  }

  // ================= FILE UPLOAD =================

  Future<void> pickFile() async {

    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf','txt','docx'],
      withData: true,
    );

    if (result == null) return;

    Uint8List fileBytes = result.files.single.bytes!;
    String fileName = result.files.single.name;

    setState(() {
      selectedFileBytes = fileBytes;
      selectedFileName = fileName;
      isLoading = true;
    });

    try {

      var request = http.MultipartRequest(
        'POST',
        Uri.parse("$baseUrl/upload-file?session_id=$sessionId"),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          fileBytes,
          filename: fileName,
        ),
      );

      var response = await request.send();

      if (response.statusCode != 200) throw Exception("Upload failed");

      setState(() {
        messages.add({
          "role": "assistant",
          "text": "📄 Document uploaded successfully. Ask questions about it."
        });
      });

    } catch (e) {

      setState(() {
        messages.add({
          "role": "assistant",
          "text": "⚠️ File upload failed."
        });
      });

    }

    setState(() => isLoading = false);
  }

  void removeFile() {
    setState(() {
      selectedFileBytes = null;
      selectedFileName = null;
    });
  }

  // ================= FEEDBACK =================

Future<void> sendFeedback(String query,String response,String type) async {

  final res = await http.post(
    Uri.parse("$baseUrl/feedback"),
    headers: {"Content-Type": "application/json"},
    body: jsonEncode({
      "query": query,
      "response": response,
      "feedback": type
    }),
  );

  final data = jsonDecode(res.body);

  setState(() {
    messages.add({
      "role": "assistant",
      "text": data["response"]
    });
  });
}

  // ================= REGENERATE =================

  void regenerateLast() {

    for (int i = messages.length - 1; i >= 0; i--) {

      if (messages[i]["role"] == "user") {
        sendMessage(messages[i]["text"]);
        break;
      }

    }
  }

  void scrollToBottom() {

    Future.delayed(const Duration(milliseconds: 300), () {

      if (scrollController.hasClients) {

        scrollController.animateTo(
          scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );

      }

    });

  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      body: Stack(
        children: [

          Positioned.fill(
            child: Image.asset("assets/kombucha_bg.png",fit: BoxFit.cover),
          ),

          Container(color: Colors.black.withOpacity(0.6)),

          Column(
            children: [

              const SizedBox(height: 50),

              Image.asset("assets/kgpt_logo.png",height: 60),

              Expanded(
                child: ListView.builder(
                  controller: scrollController,
                  padding: const EdgeInsets.all(16),
                  itemCount: messages.length,
                  itemBuilder: (context,index){

                    return ChatBubble(
                      message: messages[index],

                      onLike: (){
                        if(index>0){
                          sendFeedback(
                            messages[index-1]["text"],
                            messages[index]["text"],
                            "positive"
                          );
                        }
                      },

                      onDislike: (){
                        if(index>0){
                          sendFeedback(
                            messages[index-1]["text"],
                            messages[index]["text"],
                            "negative"
                          );
                        }
                      },

                      onCopy: (){
                        Clipboard.setData(
                          ClipboardData(text: messages[index]["text"])
                        );
                      },

                      onRegenerate: regenerateLast,

                      onSuggestionTap: (q){
                        sendMessage(q);
                      },

                    );

                  },
                ),
              ),

              if(isLoading)
                const Padding(
                  padding: EdgeInsets.all(8),
                  child: CircularProgressIndicator(),
                ),

              Container(
                padding: const EdgeInsets.all(12),
                decoration: const BoxDecoration(
                  color: Color(0xFF2C2C2C),
                  borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
                ),

                child: Row(
                  children: [

                    IconButton(
                      icon: const Icon(Icons.attach_file,color: Colors.white),
                      onPressed: pickFile,
                    ),

                    Expanded(
                      child: TextField(
                        controller: controller,
                        textInputAction: TextInputAction.send,
                        style: const TextStyle(color: Colors.white),
                        decoration: const InputDecoration(
                          hintText: "Ask anything about kombucha...",
                          hintStyle: TextStyle(color: Colors.white54),
                          border: InputBorder.none,
                        ),
                        onSubmitted: (value){
                          sendMessage();
                        },
                      ),
                    ),

                    IconButton(
                      icon: const Icon(Icons.send,color: Colors.white),
                      onPressed: ()=>sendMessage(),
                    )

                  ],
                ),
              )

            ],
          ),
        ],
      ),
    );
  }
}

class ChatBubble extends StatelessWidget {

  final Map<String,dynamic> message;

  final VoidCallback onLike;
  final VoidCallback onDislike;
  final VoidCallback onCopy;
  final VoidCallback onRegenerate;

  final Function(String) onSuggestionTap;

  const ChatBubble({
    super.key,
    required this.message,
    required this.onLike,
    required this.onDislike,
    required this.onCopy,
    required this.onRegenerate,
    required this.onSuggestionTap,
  });

  Future<void> openLink(String url) async {

    final Uri uri = Uri.parse(url);

    if(await canLaunchUrl(uri)){
      await launchUrl(
        uri,
        mode: LaunchMode.externalApplication
      );
    }

  }

  @override
  Widget build(BuildContext context){

    bool isUser = message["role"]=="user";

    return Align(
      alignment: isUser?Alignment.centerRight:Alignment.centerLeft,

      child: Container(
        margin: const EdgeInsets.symmetric(vertical:8),
        padding: const EdgeInsets.all(16),
        width: MediaQuery.of(context).size.width*0.75,

        decoration: BoxDecoration(
          color: const Color(0xFF333333),
          borderRadius: BorderRadius.circular(20),
        ),

        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            Text(
              message["text"]??"",
              style: const TextStyle(color: Colors.white,fontSize:15),
            ),

            if(!isUser &&
                message["sources"]!=null &&
                message["sources"].isNotEmpty)

              Padding(
                padding: const EdgeInsets.only(top:8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: List.generate(
                    message["sources"].length,
                        (i){

                      String citation = message["sources"][i]["citation"];
                      String link = message["sources"][i]["link"] ?? "";

                      return GestureDetector(
                        onTap: (){
                          if(link.isNotEmpty){
                            openLink(link);
                          }
                        },
                        child: Text(
                          "Source ${i+1}: $citation",
                          style: const TextStyle(
                              color: Colors.lightBlueAccent,
                              fontSize: 12,
                              decoration: TextDecoration.underline,
                              fontStyle: FontStyle.italic
                          ),
                        ),
                      );

                    },
                  ),
                ),
              ),

            if(!isUser &&
                message["suggestions"]!=null &&
                message["suggestions"].isNotEmpty)

              Padding(
                padding: const EdgeInsets.only(top:10),
                child: Wrap(
                  spacing:6,
                  runSpacing:6,

                  children: List.generate(
                    message["suggestions"].length,
                        (i){

                      return ActionChip(
                        backgroundColor: const Color(0xFF444444),

                        label: Text(
                          message["suggestions"][i],
                          style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12
                          ),
                        ),

                        onPressed: (){
                          onSuggestionTap(message["suggestions"][i]);
                        },

                      );

                    },
                  ),
                ),
              ),

            if(!isUser)
              Row(
                children: [

                  IconButton(
                    icon: const Icon(Icons.thumb_up_alt_outlined,color: Colors.white),
                    onPressed: onLike,
                  ),

                  IconButton(
                    icon: const Icon(Icons.thumb_down_alt_outlined,color: Colors.white),
                    onPressed: onDislike,
                  ),

                  IconButton(
                    icon: const Icon(Icons.copy,color: Colors.white),
                    onPressed: onCopy,
                  ),

                  IconButton(
                    icon: const Icon(Icons.refresh,color: Colors.white),
                    onPressed: onRegenerate,
                  ),

                ],
              )

          ],
        ),
      ),
    );
  }
}