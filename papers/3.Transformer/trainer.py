
def inference(model: nn.Module, iterator: BucketIterator, beam_size, alpha):
    '''
    Beam search for inference.
    $s(Y, X)=log(P(Y \rvert X))/lp(Y)$ where $lp(Y) = (5 + \rvert Y \rvert)^\alpha $
    TODO: batch처럼 들어가는 구조: [Bacth, Beam, Seq]
    '''
    model.eval() # remove drop-out layer

    # FIXME: FINAL인데 그냥 parameter로 받으면 안되나?
    pred = torch.zeros(BATCH_SIZE, beam_size, SEQ_LEN).fill_(PAD_IDX)
    pred[:, 0] = TRG.vocab.stoi['<SOS>']

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            for t in range(1, trg.size(2)):
                # iterate over max sequence length
                input_ = trg[:, :t] # [B, 1]: t-th word of each batch
                output = model(src, input_) # [B, seq_len, tgt_vocab]
                output = output[:, t] # [B, tgt_vocab]
                # $s(Y,X)=log(P(Y \rvert X)) / \frac{(5+ \rvert Y \rvert)^\alpha}{(5+1)^\alpha}$
                log_prb = nn.functional.log_softmax(pred, dim=-1)
                lp = # length penalty
                topk = torch.topk(output, beam_size)[1]
                pred[:, :, t, :] = topk

    return 