"""Small registered-output packet; never include stale files from another run."""


class DiagnosticPacket:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        freeze(self.root/'protocol.json', identity)
        self.outputs = {self.root/'protocol.json'}
        self.inputs = {}

    def track(self, path):
        path = Path(path)
        self.inputs[str(path)] = file_hash(path)
        return path

    def csv(self, name, frame):
        path = self.root/name
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        self.outputs.add(path)
        return path

    def json(self, name, value):
        path = self.root/name
        save_json(path, value)
        self.outputs.add(path)
        return path

    def text(self, name, value):
        path = self.root/name
        path.write_text(value, encoding='utf-8')
        self.outputs.add(path)
        return path

    def finish(self, summary, report, archive_name, auto_download=True):
        import zipfile
        changed = [p for p, h in self.inputs.items() if not Path(p).is_file() or file_hash(p) != h]
        summary = dict(summary, source_files_unchanged=not changed, changed_inputs=changed)
        if changed:
            summary['status'] = 'inputs_changed_results_not_comparable'
        self.json('input_manifest.json', self.inputs)
        self.json('RUN_COMPLETE.json', summary)
        self.text('REPORT_ZH.md', report)
        archive = self.root/archive_name
        temporary = archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(self.outputs):
                bundle.write(path, str(path.relative_to(self.root)))
        temporary.replace(archive)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('完整结果包：', archive)
        if auto_download:
            try:
                from google.colab import files
                files.download(str(archive))
            except Exception as error:
                print('自动下载未触发，请从以上 Drive 路径手动下载：', type(error).__name__, str(error))
        return archive


def diagnostic_show(title, frame, limit=30):
    print('\n'+title)
    if frame.empty:
        print('无记录或此项输入缺失；请查看 coverage。')
        return
    with pd.option_context('display.max_rows', limit+2, 'display.max_columns', None, 'display.max_colwidth', 90):
        display(frame.head(limit))
    if len(frame) > limit:
        print(f'这里显示前 {limit} 行；结果 ZIP 保存全部 {len(frame)} 行。')
