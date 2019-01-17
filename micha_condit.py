import torch
import numpy as np

class cond_prob:

    def __init__(self):
        pass

    def sequentialDecode(self, decoder, encoderHidden, encoderOut, targform,
                         output_vocab):
        dec_hidden = decoder._init_state(encoderHidden)

        res = 0
        for inChr, outChr in zip(targform, targform[1:]):
            xid = output_vocab.stoi[inChr]
            xidt = torch.LongTensor([xid,]).view(1, -1)
            if torch.cuda.is_available():
                xidt = xidt.cuda()

            # None is the prefix where vectors would normally be
            # TODO add support for vectors
            (predict, dec_hidden, attn) = decoder.forward_step(
                None, xidt, dec_hidden, encoderOut,
                function=torch.nn.functional.log_softmax)

            prs = predict.detach()
            # TODO check if this is correct---should be cuda()? 
            if torch.cuda.is_available():
                prs = prs.cpu()
            prs = prs.numpy().squeeze()
            #print("Size of prs", prs.shape)                                        
            maxchr = np.argmax(prs)
            oid = output_vocab.stoi[outChr]
            #print("Output:", maxchr, output_vocab.itos[maxchr], end=" ")           
            #print("Target:", outChr, prs[oid])                                     
            res += prs[oid]

        #print("Total:", res)                                                       
        return res
     
    def condPr(self, formIn, formsOut, cellIn, cellOut, model,
               input_vocab, output_vocab):
        # srcform = (["OUT=inpcell=c%d" % cellIn, "OUT=outpcell=c%d" % cellOut] +
                   # list(formIn) + ["<eos>"])

        srcform = cellOut + formIn
        src_id_seq = torch.LongTensor([input_vocab.stoi[tok]
                                       for tok in srcform]).view(1, -1)

        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        encoderOut, encoderHidden = model.encoder(src_id_seq, [len(srcform),])

        res = []
        for formOut in formsOut:
            targForm = ["<bos>"] + list(formOut) + ["<eos>"]
            targ_id_seq = torch.LongTensor([output_vocab.stoi[tok]
                                            for tok in targForm]).view(1, -1)

            if torch.cuda.is_available():
                targ_id_seq = targ_id_seq.cuda()

            pr = self.sequentialDecode(model.decoder, encoderHidden, encoderOut,
                             targForm, output_vocab)
            res.append(pr)

        return res
